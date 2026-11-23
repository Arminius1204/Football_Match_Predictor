"""
Football Match Goal-Difference Prediction with an Artificial Neural Network (PyTorch)
=====================================================================================
Deep Learning (CSC403-8) - CIA 1, Component I A

Task      : Regression. Predict the final goal difference (home goals - away goals)
            of an English Premier League match using only information available
            BEFORE kick-off.
Data      : EPL seasons 2014-15 to 2024-25 (4,180 matches), football-data.co.uk
            format, mirrored at github.com/datasets/football-datasets.
Model     : Multilayer perceptron written from explicit weight/bias parameters
            (z = xW^T + b, a = f(z)) and trained with backpropagation (Adam, MSE).
Experiments:
  E1  Activation functions  - Sigmoid vs Tanh vs ReLU vs Leaky ReLU vs ELU
                              (architecture, optimiser, data, epochs fixed)
  E2  Network configuration - single neuron (linear) up to a 3-hidden-layer MLP
                              (activation fixed to the E1 winner)
  Each configuration is trained with 5 random seeds; mean +/- std is reported.

Run:  python football_ann.py            (about 2-4 minutes on a laptop CPU)
Outputs are written to ./outputs (figures, results.json, tables as CSV).
"""

import glob
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
OUT_DIR = "outputs"
SEEDS = [0, 1, 2, 3, 4]
WINDOW = 5            # number of previous matches used for rolling form
MIN_HISTORY = 3       # a team needs at least this many previous matches
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25         # early stopping on validation loss
os.makedirs(OUT_DIR, exist_ok=True)
torch.set_num_threads(4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_matches():
    frames = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "season-*.csv"))):
        season = os.path.basename(path)[7:11]          # e.g. "1415"
        df = pd.read_csv(path)
        df["Season"] = season
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    df["GD"] = df["FTHG"] - df["FTAG"]                 # regression target
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering (strictly pre-match: no information leakage)
# ---------------------------------------------------------------------------
def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def build_features(df):
    """For each match, summarise each team's previous WINDOW matches and current Elo.

    Every statistic is computed from matches that finished BEFORE the current one,
    then the team's history is updated with the current result.
    """
    history = {}                      # team -> list of per-match dicts
    elo = {}                          # team -> rating
    K, HOME_ADV, START = 20.0, 60.0, 1500.0
    rows = []

    def form(team):
        h = history.get(team, [])[-WINDOW:]
        if len(h) < MIN_HISTORY:
            return None
        keys = ["gf", "ga", "sf", "sa", "stf", "sta", "cf", "pts"]
        return {k: float(np.mean([m[k] for m in h])) for k in keys}

    for _, m in df.iterrows():
        ht, at = m["HomeTeam"], m["AwayTeam"]
        eh, ea = elo.get(ht, START), elo.get(at, START)
        fh, fa = form(ht), form(at)
        if fh is not None and fa is not None:
            feat = {"Date": m["Date"], "Season": m["Season"], "HomeTeam": ht, "AwayTeam": at,
                    "GD": m["GD"], "FTR": m["FTR"],
                    "elo_home": eh, "elo_away": ea, "elo_diff": eh + HOME_ADV - ea}
            for k, v in fh.items():
                feat[f"home_{k}"] = v
            for k, v in fa.items():
                feat[f"away_{k}"] = v
            rows.append(feat)

        # ---- update state with the finished match ----
        hg, ag = m["FTHG"], m["FTAG"]
        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 if ag > hg else (1 if hg == ag else 0)
        history.setdefault(ht, []).append(dict(gf=hg, ga=ag, sf=m["HS"], sa=m["AS"],
                                               stf=m["HST"], sta=m["AST"], cf=m["HC"], pts=hp))
        history.setdefault(at, []).append(dict(gf=ag, ga=hg, sf=m["AS"], sa=m["HS"],
                                               stf=m["AST"], sta=m["HST"], cf=m["AC"], pts=ap))
        s_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        e_home = elo_expected(eh + HOME_ADV, ea)
        margin = np.log(abs(hg - ag) + 1) + 1          # goal-margin multiplier
        elo[ht] = eh + K * margin * (s_home - e_home)
        elo[at] = ea - K * margin * (s_home - e_home)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. ANN built from explicit neurons:  z = x W^T + b ,  a = f(z)
# ---------------------------------------------------------------------------
class NeuronLayer(nn.Module):
    """A fully connected layer of artificial neurons written out explicitly.

    Neuron j computes  z_j = sum_i w_ji * x_i + b_j , i.e. z = x W^T + b in matrix form.
    """

    def __init__(self, n_in, n_out, init="he"):
        super().__init__()
        std = np.sqrt(2.0 / n_in) if init == "he" else np.sqrt(1.0 / n_in)  # He / Xavier(LeCun)
        self.W = nn.Parameter(torch.randn(n_out, n_in) * std)
        self.b = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        return x @ self.W.T + self.b


ACTIVATIONS = {
    "Sigmoid": (nn.Sigmoid, "xavier"),
    "Tanh": (nn.Tanh, "xavier"),
    "ReLU": (nn.ReLU, "he"),
    "LeakyReLU": (lambda: nn.LeakyReLU(0.01), "he"),
    "ELU": (nn.ELU, "he"),
}


class MLP(nn.Module):
    def __init__(self, n_in, hidden, activation="ReLU", dropout=0.0):
        super().__init__()
        act_cls, init = ACTIVATIONS[activation]
        layers, prev = [], n_in
        for h in hidden:
            layers += [NeuronLayer(prev, h, init), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(NeuronLayer(prev, 1, "xavier"))  # linear output neuron (regression)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 4. Gradient check: manual backpropagation vs PyTorch autograd
# ---------------------------------------------------------------------------
def gradient_check(n_in):
    """One hidden layer (tanh), MSE loss. Derive dL/dW by hand and compare with autograd."""
    torch.manual_seed(0)
    x = torch.randn(8, n_in, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)
    W1 = torch.randn(4, n_in, dtype=torch.float64, requires_grad=True)
    b1 = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(4, dtype=torch.float64, requires_grad=True)
    b2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    z1 = x @ W1.T + b1          # pre-activation of hidden neurons
    a1 = torch.tanh(z1)         # hidden activations
    y_hat = a1 @ w2 + b2        # output neuron (linear)
    loss = ((y_hat - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        n = x.shape[0]
        d_yhat = 2 * (y_hat - y) / n              # dL/dy_hat
        g_w2 = a1.T @ d_yhat                      # dL/dw2
        g_b2 = d_yhat.sum(0, keepdim=True)
        d_a1 = d_yhat[:, None] * w2[None, :]      # dL/da1
        d_z1 = d_a1 * (1 - a1 ** 2)               # tanh'(z) = 1 - tanh(z)^2
        g_W1 = d_z1.T @ x
        g_b1 = d_z1.sum(0)
    diffs = {name: float((manual - auto).abs().max()) for name, manual, auto in
             [("W1", g_W1, W1.grad), ("b1", g_b1, b1.grad), ("w2", g_w2, w2.grad), ("b2", g_b2, b2.grad)]}
    return diffs


# ---------------------------------------------------------------------------
# 5. Training and evaluation
# ---------------------------------------------------------------------------
def gd_to_outcome(pred_gd, t):
    return np.where(pred_gd > t, "H", np.where(pred_gd < -t, "A", "D"))


def best_threshold(pred, true_ftr):
    """Choose the draw band |GD_hat| <= t that maximises W/D/L accuracy (tuned on validation only)."""
    grid = np.linspace(0.0, 1.0, 101)
    accs = [(gd_to_outcome(pred, t) == true_ftr).mean() for t in grid]
    return float(grid[int(np.argmax(accs))])


def train_model(data, hidden, activation, seed, dropout=0.0, record_curves=False):
    set_seed(seed)
    Xtr, ytr, Xva, yva = data["Xtr"], data["ytr"], data["Xva"], data["yva"]
    model = MLP(Xtr.shape[1], hidden, activation, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    n = len(Xtr)
    best_val, best_state, best_epoch, wait = np.inf, None, 0, 0
    curves = {"train": [], "val": [], "grad_norm_first_layer": []}
    g = torch.Generator().manual_seed(seed)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g)
        ep_loss, gnorms = 0.0, []
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()                               # backpropagation
            gnorms.append(model.net[0].W.grad.norm().item())
            opt.step()                                    # W <- W - lr * Adam(dL/dW)
            ep_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        curves["train"].append(ep_loss / n)
        curves["val"].append(val_loss)
        curves["grad_norm_first_layer"].append(float(np.mean(gnorms)))
        if val_loss < best_val - 1e-5:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_epoch + 1, (curves if record_curves else None)


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        pva = model(data["Xva"]).numpy()
        pte = model(data["Xte"]).numpy()
    yte = data["yte"].numpy()
    t = best_threshold(pva, data["ftr_va"])
    outcome = gd_to_outcome(pte, t)
    return {
        "val_MAE": mean_absolute_error(data["yva"].numpy(), pva),
        "MAE": mean_absolute_error(yte, pte),
        "RMSE": float(np.sqrt(mean_squared_error(yte, pte))),
        "R2": r2_score(yte, pte),
        "WDL_acc": float((outcome == data["ftr_te"]).mean()),
        "draw_threshold": t,
    }, pte, outcome


def run_config(data, name, hidden, activation, dropout=0.0):
    per_seed, epochs, curves, t0 = [], [], None, time.time()
    for s in SEEDS:
        model, ep, c = train_model(data, hidden, activation, s, dropout, record_curves=(s == SEEDS[0]))
        m, pte, outcome = evaluate(model, data)
        per_seed.append(m)
        epochs.append(ep)
        if c is not None:
            curves, first_pred, first_outcome = c, pte, outcome
    agg = {k: (float(np.mean([m[k] for m in per_seed])), float(np.std([m[k] for m in per_seed])))
           for k in ["val_MAE", "MAE", "RMSE", "R2", "WDL_acc"]}
    agg["best_epoch"] = (float(np.mean(epochs)), float(np.std(epochs)))
    agg["params"] = count_params(MLP(data["Xtr"].shape[1], hidden, activation))
    agg["time_s"] = time.time() - t0
    print(f"  {name:<28} valMAE {agg['val_MAE'][0]:.3f}  MAE {agg['MAE'][0]:.3f}±{agg['MAE'][1]:.3f}  RMSE {agg['RMSE'][0]:.3f}  "
          f"R2 {agg['R2'][0]:.3f}±{agg['R2'][1]:.3f}  WDL {agg['WDL_acc'][0]:.3f}  "
          f"epoch {agg['best_epoch'][0]:.0f}")
    return {"name": name, "hidden": hidden, "activation": activation, "dropout": dropout,
            "metrics": agg, "curves": curves, "pred": first_pred, "outcome": first_outcome}


def fmt(ms):
    return f"{ms[0]:.3f} ± {ms[1]:.3f}"


def table(results):
    return pd.DataFrame([{
        "Model": r["name"], "Params": r["metrics"]["params"], "Val MAE": fmt(r["metrics"]["val_MAE"]),
        "MAE": fmt(r["metrics"]["MAE"]), "RMSE": fmt(r["metrics"]["RMSE"]),
        "R2": fmt(r["metrics"]["R2"]), "W/D/L acc": fmt(r["metrics"]["WDL_acc"]),
        "Best epoch": f"{r['metrics']['best_epoch'][0]:.0f}"} for r in results])


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_matches()
    feats = build_features(df)
    feature_cols = [c for c in feats.columns
                    if c.startswith(("home_", "away_", "elo_")) and c not in ("HomeTeam", "AwayTeam")]
    print(f"Matches loaded: {len(df)}   usable after history filter: {len(feats)}   "
          f"features: {len(feature_cols)}")

    # Chronological split (no shuffling across time -> no look-ahead)
    train_seasons = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122"]
    tr = feats[feats.Season.isin(train_seasons)]
    va = feats[feats.Season == "2223"]
    te = feats[feats.Season.isin(["2324", "2425"])]
    print(f"Train {len(tr)} | Validation {len(va)} | Test {len(te)}")

    scaler = StandardScaler().fit(tr[feature_cols])   # fit on training data only
    to_t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    data = {
        "Xtr": to_t(scaler.transform(tr[feature_cols])), "ytr": to_t(tr.GD),
        "Xva": to_t(scaler.transform(va[feature_cols])), "yva": to_t(va.GD),
        "Xte": to_t(scaler.transform(te[feature_cols])), "yte": to_t(te.GD),
        "ftr_va": va.FTR.values, "ftr_te": te.FTR.values,
    }

    results = {"dataset": {"matches": len(df), "usable": len(feats), "features": feature_cols,
                           "train": len(tr), "val": len(va), "test": len(te),
                           "gd_mean_train": float(tr.GD.mean()), "gd_std_train": float(tr.GD.std()),
                           "outcome_share_test": te.FTR.value_counts(normalize=True).round(4).to_dict()}}

    # --- Gradient check ---
    diffs = gradient_check(len(feature_cols))
    print("Gradient check (max |manual - autograd|):", {k: f"{v:.2e}" for k, v in diffs.items()})
    results["gradient_check"] = diffs

    # --- Baselines ---
    yte = data["yte"].numpy()
    mean_pred = np.full_like(yte, data["ytr"].mean().item())
    base = {"MAE": mean_absolute_error(yte, mean_pred),
            "RMSE": float(np.sqrt(mean_squared_error(yte, mean_pred))),
            "R2": r2_score(yte, mean_pred),
            "WDL_acc_always_home": float((data["ftr_te"] == "H").mean())}
    print("Baseline (predict training mean GD):", {k: round(v, 3) for k, v in base.items()})
    results["baseline_mean"] = base

    # --- Experiment 1: activation functions (architecture fixed at 64-32) ---
    print("\nE1: activation functions, architecture [64, 32]")
    e1 = [run_config(data, a, [64, 32], a) for a in ACTIVATIONS]
    best_act = min(e1, key=lambda r: r["metrics"]["val_MAE"][0])["activation"]   # select on validation only
    print("Best activation by validation MAE:", best_act)

    # --- Experiment 2: network configuration (activation fixed) ---
    print(f"\nE2: network configurations, activation {best_act}")
    configs = [("Single neuron (linear)", [], 0.0),
               ("1 hidden [16]", [16], 0.0),
               ("1 hidden [64]", [64], 0.0),
               ("2 hidden [64, 32]", [64, 32], 0.0),
               ("3 hidden [128, 64, 32]", [128, 64, 32], 0.0),
               ("3 hidden + dropout 0.3", [128, 64, 32], 0.3)]
    e2 = [run_config(data, n, h, best_act, d) for n, h, d in configs]

    strip = lambda rs: [{k: v for k, v in r.items() if k not in ("curves", "pred", "outcome")} for r in rs]
    results["E1"], results["E2"], results["best_activation"] = strip(e1), strip(e2), best_act
    t1, t2 = table(e1), table(e2)
    t1.to_csv(f"{OUT_DIR}/table_E1_activations.csv", index=False)
    t2.to_csv(f"{OUT_DIR}/table_E2_architectures.csv", index=False)
    print("\n", t1.to_string(index=False), "\n\n", t2.to_string(index=False))

    # -------------------------------------------------------------------
    # 7. Figures
    # -------------------------------------------------------------------
    colors = {"Sigmoid": "#d62728", "Tanh": "#ff7f0e", "ReLU": "#1f77b4",
              "LeakyReLU": "#2ca02c", "ELU": "#9467bd"}

    # Fig 1: activation functions and their derivatives
    z = torch.linspace(-5, 5, 400, requires_grad=True)
    _, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, (cls, _) in ACTIVATIONS.items():
        a = cls()(z)
        (g,) = torch.autograd.grad(a.sum(), z)
        ls = {"ReLU": "-", "LeakyReLU": "--", "ELU": ":"}.get(name, "-")
        ax[0].plot(z.detach(), a.detach(), label=name, color=colors[name], ls=ls, lw=2)
        ax[1].plot(z.detach(), g, label=name, color=colors[name], ls=ls, lw=2)
    ax[0].set_title("Activation f(z)"); ax[1].set_title("Derivative f'(z)")
    for a_ in ax:
        a_.axhline(0, color="grey", lw=0.5); a_.axvline(0, color="grey", lw=0.5)
        a_.set_xlabel("z"); a_.legend(fontsize=8)
    ax[0].set_ylim(-1.5, 3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_activations.png", dpi=150); plt.close()

    # Fig 2: training / validation loss curves per activation (seed 0)
    _, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in e1:
        c = colors[r["activation"]]
        ax[0].plot(r["curves"]["train"], color=c, label=r["name"])
        ax[1].plot(r["curves"]["val"], color=c, label=r["name"])
    ax[0].set_title("Training loss (MSE)"); ax[1].set_title("Validation loss (MSE)")
    for a_ in ax:
        a_.set_xlabel("Epoch"); a_.legend(fontsize=8)
    ax[0].set_ylim(2.2, 4.0); ax[1].set_ylim(2.3, 3.6)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig2_loss_curves_activations.png", dpi=150); plt.close()

    # Fig 3: first-layer gradient norm per activation
    _, ax = plt.subplots(figsize=(6.5, 4))
    for r in e1:
        ax.plot(r["curves"]["grad_norm_first_layer"], color=colors[r["activation"]], label=r["name"])
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Mean ||dL/dW1||")
    ax.set_title("Gradient magnitude reaching the first layer"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig3_gradient_norms.png", dpi=150); plt.close()

    # Fig 4: E1 and E2 bar charts (MAE and R2 with std error bars)
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    names = [r["name"] for r in e1]
    ax[0].bar(names, [r["metrics"]["MAE"][0] for r in e1], yerr=[r["metrics"]["MAE"][1] for r in e1],
              color=[colors[n] for n in names], capsize=4)
    ax[0].axhline(base["MAE"], ls="--", color="k", label="Mean baseline")
    ax[0].set_ylim(1.2, 1.5); ax[0].set_title("E1: Test MAE by activation (5 seeds)"); ax[0].legend()
    names2 = [r["name"] for r in e2]
    ax[1].bar(range(len(e2)), [r["metrics"]["R2"][0] for r in e2],
              yerr=[r["metrics"]["R2"][1] for r in e2], color="#1f77b4", capsize=4)
    ax[1].set_xticks(range(len(e2))); ax[1].set_xticklabels(names2, rotation=25, ha="right", fontsize=8)
    ax[1].set_ylim(0.18, 0.23); ax[1].set_title(f"E2: Test R² by configuration ({best_act})")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4_experiment_bars.png", dpi=150); plt.close()

    # Fig 6: predicted vs actual GD, and W/D/L confusion matrix for best overall model
    best = min(e1 + e2, key=lambda r: r["metrics"]["val_MAE"][0])          # select on validation only
    results["best_overall"] = best["name"] + f" ({best['activation']}, {best['hidden']})"
    _, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(yte))
    ax[0].scatter(best["pred"], yte + jitter, s=8, alpha=0.35)
    lim = [-6, 6]; ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlim(-3, 3); ax[0].set_ylim(-6.5, 6.5)
    ax[0].set_xlabel("Predicted goal difference"); ax[0].set_ylabel("Actual goal difference (jittered)")
    ax[0].set_title(f"Predicted vs actual (test) - {best['name']}")
    labels = ["H", "D", "A"]
    cm = confusion_matrix(data["ftr_te"], best["outcome"], labels=labels)
    ax[1].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax[1].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["Home win", "Draw", "Away win"])
    ax[1].set_yticks(range(3)); ax[1].set_yticklabels(["Home win", "Draw", "Away win"])
    ax[1].set_xlabel("Predicted (from GD)"); ax[1].set_ylabel("Actual")
    ax[1].set_title("Match outcome derived from predicted GD")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig6_best_model.png", dpi=150); plt.close()
    results["best_confusion_matrix"] = cm.tolist()

    # Fig 5: E2 loss curves (overfitting view)
    _, ax = plt.subplots(figsize=(6.5, 4))
    for r in e2:
        ax.plot(r["curves"]["val"], label=r["name"])
    ax.set_ylim(2.3, 3.6); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MSE")
    ax.set_title("E2: validation loss by configuration"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig5_loss_curves_architectures.png", dpi=150); plt.close()

    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nBest overall:", results["best_overall"])
    print(f"Saved figures and results to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
