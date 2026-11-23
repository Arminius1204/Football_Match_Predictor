"""NumPy-only inference: everything the deployed app needs at runtime.

This module imports only numpy and the standard library: no torch, scikit-learn, scipy
or football_ann. The training side (features.py, model.py, evaluation.py) imports the
shared helpers from here, so each piece of logic exists once.

The forward pass mirrors football_ann.MLP: each hidden layer computes
a = f(x W^T + b), and the output neuron is linear. Weights come from
artifacts/weights.npz, exported from the trained PyTorch models by train_app_model.py.
tests/test_numpy_inference.py checks NumPy == PyTorch within 1e-5 for every team pair.
"""

import math

import numpy as np

# Same functions as football_ann.ACTIVATIONS (LeakyReLU slope 0.01, ELU alpha 1).
ACTIVATIONS = {
    "Sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "Tanh": np.tanh,
    "ReLU": lambda z: np.maximum(z, 0.0),
    "LeakyReLU": lambda z: np.where(z > 0, z, 0.01 * z),
    "ELU": lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0.0))),
}


# ---------------------------------------------------------------------------
# Features and scaling
# ---------------------------------------------------------------------------
def compose_features(home_snap, away_snap, home_adv, feature_cols):
    """Build the model's feature vector for (home, away) from two team snapshots.

    Mirrors the row football_ann.build_features writes for a match: elo_home,
    elo_away, elo_diff = elo_home + home_adv - elo_away, then home_<k> / away_<k>
    form means. Returns a 1-D float64 array ordered like ``feature_cols``.
    """
    values = {"elo_home": home_snap["elo"], "elo_away": away_snap["elo"],
              "elo_diff": home_snap["elo"] + home_adv - away_snap["elo"]}
    for k, v in home_snap["form"].items():
        values[f"home_{k}"] = v
    for k, v in away_snap["form"].items():
        values[f"away_{k}"] = v
    return np.array([values[c] for c in feature_cols], dtype=np.float64)


def transform(X, mean, scale):
    """Same arithmetic as StandardScaler.transform (float64), for 2-D or 1-D input."""
    return (np.asarray(X, dtype=np.float64) - mean) / scale


# ---------------------------------------------------------------------------
# Ensemble forward pass
# ---------------------------------------------------------------------------
def load_numpy_ensemble(path):
    """Read weights.npz -> (members, activation).

    Each member is a list of (W, b) pairs, one per layer, in forward order.
    """
    with np.load(path, allow_pickle=False) as z:
        activation = str(z["activation"])
        members = []
        for i in range(int(z["n_models"])):
            n_layers = int(z[f"n_layers_{i}"])
            members.append([(z[f"m{i}_W{j}"].astype(np.float64), z[f"m{i}_b{j}"].astype(np.float64))
                            for j in range(n_layers)])
    if activation not in ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}")
    return members, activation


def mlp_forward(X, layers, activation):
    """One network: hidden layers a = f(x W^T + b), then a linear output neuron."""
    a = np.asarray(X, dtype=np.float64)
    f = ACTIVATIONS[activation]
    for W, b in layers[:-1]:
        a = f(a @ W.T + b)
    W, b = layers[-1]
    return (a @ W.T + b)[..., 0]


def ensemble_members(members, activation, X):
    """Per-member predictions, shape (n_models, n_samples)."""
    return np.stack([mlp_forward(X, layers, activation) for layers in members])


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------
def gd_to_outcome(pred_gd, t):
    """W/D/L from predicted GD with draw band |GD_hat| <= t.

    Same rule as football_ann.gd_to_outcome. Repeated here so the app doesn't import
    football_ann (and through it torch and matplotlib). A test checks they agree.
    """
    pred_gd = np.asarray(pred_gd)
    return np.where(pred_gd > t, "H", np.where(pred_gd < -t, "A", "D"))


def assign_band(pred, cutoffs):
    """Band index per prediction: 0 below the first cutoff, 1 between, ..., by |GD_hat|."""
    return np.digitize(np.abs(np.asarray(pred, dtype=np.float64)), cutoffs)


def lean_side(pred):
    """The side the model leans toward: 'H' if GD_hat > 0 else 'A'."""
    return np.where(np.asarray(pred) > 0, "H", "A")


_erf = np.vectorize(math.erf, otypes=[np.float64])


def normal_cdf(x):
    """Standard normal CDF via math.erf (avoids a scipy dependency at runtime)."""
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))


def outcome_probabilities(mu, sigma):
    """W/D/L probabilities under GD ~ Normal(mu, sigma). Returns shape (n, 3) in H, D, A order.

    P(home) = P(GD > 0.5), P(away) = P(GD < -0.5), P(draw) = the remainder.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    p_away = normal_cdf((-0.5 - mu) / sigma)
    p_home = 1.0 - normal_cdf((0.5 - mu) / sigma)
    p_draw = 1.0 - p_home - p_away
    return np.column_stack([p_home, p_draw, p_away])
