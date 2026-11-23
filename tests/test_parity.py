"""Parity with the submitted assignment: the app pipeline must reproduce its numbers.

README.md reports 5-seed MEANS for a few configurations (not the [128, 64, 32] model and
not single seeds), so parity is checked at two levels:
  1. The app config's per-seed models reproduce results.json mean ± std exactly.
  2. The app's split + scaler + football_ann.train_model reproduce the README table.
"""

import json

import numpy as np
import pytest

import football_ann as fa
from src import config
from src.model import train_ensemble

RESULTS = json.loads(config.ASSIGNMENT_RESULTS.read_text())


def _mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def test_app_settings_match_assignment():
    """config.SEEDS is written out (to keep the app torch-free); it must equal the assignment's."""
    assert config.SEEDS == list(fa.SEEDS)


def test_split_and_features_match_results_json(prepared, cols, matches, feats):
    d = RESULTS["dataset"]
    _, _, (tr, va, te) = prepared
    assert (len(matches), len(feats)) == (d["matches"], d["usable"])
    assert (len(tr), len(va), len(te)) == (d["train"], d["val"], d["test"])
    assert cols == d["features"]


def test_app_config_reproduces_results_json(prepared, ensemble):
    """Seeds 0-4 of [128, 64, 32] Sigmoid give the same mean ± std as the submitted run."""
    data = prepared[0]
    models, epochs = ensemble
    per_seed = [fa.evaluate(m, data)[0] for m in models]
    entry = next(e for e in RESULTS["E2"] if e["name"] == config.APP_CONFIG_NAME)
    assert entry["activation"] == config.APP_ACTIVATION and entry["hidden"] == config.APP_HIDDEN
    for k in ["val_MAE", "MAE", "RMSE", "R2", "WDL_acc"]:
        np.testing.assert_allclose(_mean_std([m[k] for m in per_seed]), entry["metrics"][k],
                                   atol=1e-6, err_msg=k)
    np.testing.assert_allclose(_mean_std(epochs), entry["metrics"]["best_epoch"])


# README "Key results" table: (hidden, activation) -> (test MAE, test R²), 5-seed means.
README_ROWS = [
    ([], "Sigmoid", 1.359, 0.217),            # single linear neuron (no activation used)
    ([16], "Sigmoid", 1.350, 0.218),
    ([64, 32], "Sigmoid", 1.353, 0.214),
    ([64, 32], "ReLU", 1.371, 0.190),
]


@pytest.mark.parametrize("hidden,activation,mae,r2", README_ROWS)
def test_readme_table_reproduces(prepared, hidden, activation, mae, r2):
    data = prepared[0]
    models, _ = train_ensemble(data, hidden, activation)
    per_seed = [fa.evaluate(m, data)[0] for m in models]
    assert round(np.mean([m["MAE"] for m in per_seed]), 3) == mae
    assert round(np.mean([m["R2"] for m in per_seed]), 3) == r2


def test_readme_mean_baseline(prepared):
    data = prepared[0]
    yte = data["yte"].numpy()
    pred = np.full_like(yte, data["ytr"].mean().item())
    assert round(float(np.mean(np.abs(yte - pred))), 3) == 1.531
