"""Fixed seeds -> identical models; saved artifacts -> identical predictions."""

import json

import numpy as np
import torch

import football_ann as fa
from src import evaluation as ev
from src.model import load_models, load_scaler, predict_ensemble, transform


def test_same_seed_same_weights(prepared):
    data = prepared[0]
    m1, _, _ = fa.train_model(data, [128, 64, 32], "Sigmoid", seed=0)
    m2, _, _ = fa.train_model(data, [128, 64, 32], "Sigmoid", seed=0)
    for (k, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k


def test_saved_models_equal_fresh_training(prepared, ensemble, artifacts_ready):
    saved, _ = load_models(artifacts_ready / "model.pt")
    for fresh, old in zip(ensemble[0], saved):
        for k, v in fresh.state_dict().items():
            assert torch.equal(v, old.state_dict()[k]), k


def test_saved_scaler_reproduces_sklearn(prepared, cols, artifacts_ready):
    data, scaler, (_, _, te) = prepared
    saved_cols, mean, scale = load_scaler(artifacts_ready / "scaler.json")
    assert saved_cols == cols
    np.testing.assert_allclose(transform(te[cols], mean, scale), scaler.transform(te[cols]),
                               rtol=0, atol=1e-12)


def test_meta_ensemble_metrics_recompute(prepared, artifacts_ready):
    """The ensemble test metrics shown in the app are recomputable from the artifacts."""
    data = prepared[0]
    models, _ = load_models(artifacts_ready / "model.pt")
    meta = json.loads((artifacts_ready / "meta.json").read_text())
    pte = predict_ensemble(models, data["Xte"])
    got = ev.regression_metrics(data["yte"].numpy(), pte)
    for k, v in got.items():
        assert np.isclose(meta["ensemble_test_metrics"][k], v, atol=1e-9), k
    acc = ev.wdl_accuracy(pte, data["ftr_te"], meta["draw_threshold"])
    assert np.isclose(meta["ensemble_test_metrics"]["WDL_acc"], acc)


def test_predictor_end_to_end(predictor, artifacts_ready):
    """Predictor (NumPy) matches the PyTorch pipeline on model.pt and is deterministic."""
    p1 = predictor.predict("Arsenal", "Chelsea")
    p2 = predictor.predict("Arsenal", "Chelsea")
    assert p1.gd == p2.gd
    x = transform(predictor.features_for("Arsenal", "Chelsea")[None, :], predictor.mean, predictor.scale)
    torch_models, _ = load_models(artifacts_ready / "model.pt")
    assert abs(p1.gd - predict_ensemble(torch_models, x)[0]) < 1e-5
    assert np.isclose(p1.gd, np.mean(p1.gd_members))
    assert p1.outcome in {"H", "D", "A"}
    assert len(predictor.teams()) == 20
