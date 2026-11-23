"""Split, scale, train and persist the app's ensemble, reusing football_ann's model code.

Training calls football_ann.train_model, so architecture, initialisation, optimiser,
early stopping and seeding are exactly the assignment's. The only code repeated from
football_ann.main() is the season split and scaler fit, because they are local to main().
tests/test_parity.py checks that they reproduce the assignment's results.
"""

import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import football_ann as fa
from src import config
from src.inference import transform  # noqa: F401  (re-exported; defined in the runtime module)


# ---------------------------------------------------------------------------
# Chronological split + scaling (as in football_ann.main)
# ---------------------------------------------------------------------------
def chronological_split(feats):
    """Train 2014-22, validation 2022-23, test 2023-25. No shuffling across time."""
    tr = feats[feats.Season.isin(config.TRAIN_SEASONS)]
    va = feats[feats.Season.isin(config.VAL_SEASONS)]
    te = feats[feats.Season.isin(config.TEST_SEASONS)]
    return tr, va, te


def prepare_data(feats, feature_cols):
    """Scale features (scaler fit on TRAIN only) and pack tensors the way main() does.

    Returns (data, scaler, (tr, va, te)), where ``data`` is the dict that
    football_ann.train_model / evaluate expect.
    """
    tr, va, te = chronological_split(feats)
    scaler = StandardScaler().fit(tr[feature_cols])          # fit on training data only
    to_t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    data = {
        "Xtr": to_t(scaler.transform(tr[feature_cols])), "ytr": to_t(tr.GD),
        "Xva": to_t(scaler.transform(va[feature_cols])), "yva": to_t(va.GD),
        "Xte": to_t(scaler.transform(te[feature_cols])), "yte": to_t(te.GD),
        "ftr_va": va.FTR.values, "ftr_te": te.FTR.values,
    }
    return data, scaler, (tr, va, te)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def train_ensemble(data, hidden=None, activation=None, seeds=None):
    """Train one model per seed with football_ann.train_model.

    Returns (models, best_epochs). Each seed is trained exactly as in the assignment's
    run_config, so model k equals the assignment's seed-k model for this configuration.
    """
    hidden = config.APP_HIDDEN if hidden is None else hidden
    activation = config.APP_ACTIVATION if activation is None else activation
    seeds = config.SEEDS if seeds is None else seeds
    models, epochs = [], []
    for s in seeds:
        model, best_epoch, _ = fa.train_model(data, hidden, activation, s)
        model.eval()
        models.append(model)
        epochs.append(best_epoch)
    return models, epochs


def predict_members(models, X):
    """Per-member predictions, shape (n_models, n_samples), as float64 numpy."""
    X = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return np.stack([m(X).numpy().astype(np.float64) for m in models])


def predict_ensemble(models, X):
    """Ensemble prediction = mean of the members' predicted goal differences."""
    return predict_members(models, X).mean(axis=0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_models(models, n_in, hidden, activation, seeds, path):
    """Save the ensemble's weights plus everything needed to rebuild the MLPs."""
    torch.save({"n_in": n_in, "hidden": list(hidden), "activation": activation,
                "seeds": list(seeds), "state_dicts": [m.state_dict() for m in models]}, path)


def load_models(path):
    """Rebuild the ensemble from save_models() output (football_ann.MLP, eval mode)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    models = []
    for sd in blob["state_dicts"]:
        m = fa.MLP(blob["n_in"], blob["hidden"], blob["activation"])
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, blob


def save_scaler(scaler, feature_cols, path):
    """Store the train-fitted scaler as plain JSON (column order matters)."""
    with open(path, "w") as f:
        json.dump({"feature_cols": list(feature_cols),
                   "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, indent=2)


def load_scaler(path):
    """Return (feature_cols, mean, scale) for transform(): (x - mean) / scale."""
    with open(path) as f:
        s = json.load(f)
    return s["feature_cols"], np.array(s["mean"]), np.array(s["scale"])


def export_numpy_weights(models, activation, path):
    """Write the ensemble's weights to .npz for the torch-free runtime (src/inference.py).

    Layout: activation, n_models, n_layers_<i>, and m<i>_W<j> / m<i>_b<j> for model i,
    layer j in forward order. Only plain arrays are stored (loadable with allow_pickle=False).
    """
    arrays = {"activation": np.array(activation), "n_models": np.array(len(models))}
    for i, m in enumerate(models):
        assert not any(isinstance(mod, torch.nn.Dropout) for mod in m.net), "dropout not exported"
        layers = [mod for mod in m.net if isinstance(mod, fa.NeuronLayer)]
        arrays[f"n_layers_{i}"] = np.array(len(layers))
        for j, layer in enumerate(layers):
            arrays[f"m{i}_W{j}"] = layer.W.detach().cpu().numpy()
            arrays[f"m{i}_b{j}"] = layer.b.detach().cpu().numpy()
    np.savez(path, **arrays)
