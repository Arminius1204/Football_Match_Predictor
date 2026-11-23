"""Metrics for the ensemble: regression error, W/D/L accuracy and probability quality.

Probabilities come from a simple predictive distribution:
    GD ~ Normal(mu = ensemble prediction, sigma = std of VALIDATION residuals)
    P(home) = P(GD > 0.5),  P(draw) = P(|GD| < 0.5),  P(away) = P(GD < -0.5)
The ±0.5 cut-offs treat the integer goal difference as a rounded continuous value.
sigma is estimated on validation only, so the test set stays untouched until scoring.
"""

import numpy as np
from scipy.special import ndtr          # standard normal CDF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import football_ann as fa
# Shared with the app's NumPy-only runtime; defined once in src/inference.py.
from src.inference import assign_band, lean_side, outcome_probabilities  # noqa: F401

OUTCOMES = ["H", "D", "A"]              # column order of every probability array


# ---------------------------------------------------------------------------
# Point predictions
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    """MAE, RMSE and R² of goal-difference predictions."""
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def wdl_accuracy(pred_gd, true_ftr, threshold):
    """Accuracy of the assignment's W/D/L rule: draw if |GD_hat| <= threshold."""
    return float((fa.gd_to_outcome(pred_gd, threshold) == true_ftr).mean())


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------
def residual_sigma(y_true, y_pred):
    """Std of residuals (y - y_hat). Called on validation data only."""
    return float(np.std(np.asarray(y_true) - np.asarray(y_pred)))


def one_hot(ftr):
    """Encode actual results ('H'/'D'/'A') as an (n, 3) indicator matrix."""
    ftr = np.asarray(ftr)
    return np.column_stack([(ftr == o).astype(float) for o in OUTCOMES])


def log_loss(probs, ftr, eps=1e-15):
    """Mean negative log-probability of the actual result (lower is better)."""
    p = np.clip(np.asarray(probs), eps, 1.0)
    return float(-np.mean(np.sum(one_hot(ftr) * np.log(p), axis=1)))


def base_rates(ftr):
    """Outcome frequencies in H, D, A order. Fit on TRAINING results for the baseline."""
    return one_hot(ftr).mean(axis=0)


def calibration_table(probs, ftr, n_bins=10):
    """Reliability table per outcome: in each probability bin, mean predicted vs observed.

    Fixed-width bins [0, 0.1), ..., [0.9, 1.0]. Empty bins are left out. Returns
    {outcome: [{"bin", "n", "mean_pred", "observed", "gap"}...]}.
    """
    y = one_hot(ftr)
    edges = np.linspace(0, 1, n_bins + 1)
    table = {}
    for j, o in enumerate(OUTCOMES):
        p = probs[:, j]
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        rows = []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() == 0:
                continue
            mp, ob = float(p[mask].mean()), float(y[mask, j].mean())
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                         "mean_pred": mp, "observed": ob, "gap": ob - mp})
        table[o] = rows
    return table


def expected_calibration_error(table, n_total):
    """ECE per outcome: count-weighted mean |observed - predicted| over the bins."""
    return {o: float(sum(r["n"] * abs(r["gap"]) for r in rows) / n_total)
            for o, rows in table.items()}


# ---------------------------------------------------------------------------
# Wording bands for the plain readout
# ---------------------------------------------------------------------------
def band_cutoffs(pred_val, quantiles):
    """Cutoffs on |GD_hat| from VALIDATION predictions only (quantiles of their size)."""
    return [float(np.quantile(np.abs(pred_val), q)) for q in quantiles]


def band_report(pred, ftr, cutoffs, names, min_gap, max_p):
    """Per band on held-out data: how often the leaned-to side actually won.

    A draw counts as "not won". Then the pre-set rule is applied: keep the bands only if
    the top band beats the bottom band by >= min_gap with one-sided z-test p < max_p.
    """
    band = assign_band(pred, cutoffs)
    won = lean_side(pred) == np.asarray(ftr)
    ftr = np.asarray(ftr)
    rows = []
    edges = [0.0] + list(cutoffs) + [None]
    for b, name in enumerate(names):
        m = band == b
        n = int(m.sum())
        rows.append({"band": name, "abs_gd_from": edges[b], "abs_gd_to": edges[b + 1], "n": n,
                     "lean_won": float(won[m].mean()) if n else float("nan"),
                     "draw": float((ftr[m] == "D").mean()) if n else float("nan"),
                     "lean_lost": float((~won[m] & (ftr[m] != "D")).mean()) if n else float("nan")})
    lo, hi = rows[0], rows[-1]
    gap = hi["lean_won"] - lo["lean_won"]
    # One-sided two-proportion z-test (pooled), H1: top band wins more often.
    k1, k2 = hi["lean_won"] * hi["n"], lo["lean_won"] * lo["n"]
    pooled = (k1 + k2) / (hi["n"] + lo["n"])
    se = np.sqrt(pooled * (1 - pooled) * (1 / hi["n"] + 1 / lo["n"]))
    z = gap / se
    p = float(1 - ndtr(z))
    keep = bool(gap >= min_gap and p < max_p)
    reason = (f"'{hi['band']}' lean won {hi['lean_won']:.1%} vs '{lo['band']}' {lo['lean_won']:.1%} "
              f"(gap {gap:+.1%}, z = {z:.2f}, one-sided p = {p:.2g}); rule: gap >= {min_gap:.0%} and p < {max_p}")
    return {"cutoffs": cutoffs, "rows": rows, "gap": float(gap), "z": float(z), "p_value": p,
            "show_bands": keep, "reason": reason}


def probability_report(probs, ftr, baseline_probs, max_ece):
    """Score the probabilities against a base-rate baseline and apply the display rule.

    Show the probabilities only if (a) model log loss < baseline log loss and
    (b) every outcome's ECE <= max_ece. Returns a JSON-ready dict including the
    decision and a plain-English reason.
    """
    n = len(ftr)
    base = np.tile(baseline_probs, (n, 1))
    table = calibration_table(probs, ftr)
    ece = expected_calibration_error(table, n)
    ll_model, ll_base = log_loss(probs, ftr), log_loss(base, ftr)
    brier = lambda p: float(np.mean(np.sum((p - one_hot(ftr)) ** 2, axis=1)))

    beats = ll_model < ll_base
    calibrated = all(v <= max_ece for v in ece.values())
    if beats and calibrated:
        reason = (f"log loss {ll_model:.4f} < baseline {ll_base:.4f} and max ECE "
                  f"{max(ece.values()):.3f} <= {max_ece}")
    else:
        parts = []
        if not beats:
            parts.append(f"log loss {ll_model:.4f} does not beat baseline {ll_base:.4f}")
        if not calibrated:
            worst = max(ece, key=ece.get)
            parts.append(f"ECE for {worst} is {ece[worst]:.3f} > {max_ece}")
        reason = "; ".join(parts)
    return {"log_loss_model": ll_model, "log_loss_baseline": ll_base,
            "brier_model": brier(probs), "brier_baseline": brier(base),
            "baseline_probs": dict(zip(OUTCOMES, map(float, baseline_probs))),
            "mean_pred_probs": dict(zip(OUTCOMES, map(float, probs.mean(axis=0)))),
            "observed_rates": dict(zip(OUTCOMES, map(float, one_hot(ftr).mean(axis=0)))),
            "ece": ece, "max_ece_allowed": max_ece, "calibration": table,
            "show_probabilities": bool(beats and calibrated), "reason": reason}
