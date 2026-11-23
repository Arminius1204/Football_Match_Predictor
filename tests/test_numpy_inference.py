"""The torch-free runtime must match PyTorch exactly enough, and the app must stay torch-free."""

import itertools
import subprocess
import sys

import numpy as np
from scipy.special import ndtr

import football_ann as fa
from src import inference
from src.model import load_models, predict_members

TOL = 1e-5


def test_numpy_matches_pytorch_for_every_pair(predictor, artifacts_ready):
    """All 1,190 ordered pairs (35 teams): every seed and the ensemble agree within 1e-5."""
    teams = predictor.teams(include_all=True)
    pairs = list(itertools.permutations(teams, 2))
    assert len(pairs) == 1190
    X_raw = np.stack([predictor.features_for(h, a) for h, a in pairs])

    numpy_members = predictor.member_predictions(X_raw)
    torch_models, _ = load_models(artifacts_ready / "model.pt")
    torch_members = predict_members(torch_models, inference.transform(X_raw, predictor.mean, predictor.scale))

    assert numpy_members.shape == torch_members.shape == (5, 1190)
    worst = np.abs(numpy_members - torch_members).max()
    assert worst < TOL, f"max |numpy - torch| per seed = {worst:.2e}"
    assert np.abs(numpy_members.mean(0) - torch_members.mean(0)).max() < TOL


def test_weights_npz_equals_model_pt(artifacts_ready):
    """The exported arrays are the PyTorch parameters, layer by layer."""
    members, activation = inference.load_numpy_ensemble(artifacts_ready / "weights.npz")
    torch_models, blob = load_models(artifacts_ready / "model.pt")
    assert activation == blob["activation"] and len(members) == len(torch_models)
    for layers, m in zip(members, torch_models):
        neurons = [mod for mod in m.net if isinstance(mod, fa.NeuronLayer)]
        assert len(layers) == len(neurons)
        for (W, b), n in zip(layers, neurons):
            np.testing.assert_array_equal(W, n.W.detach().numpy().astype(np.float64))
            np.testing.assert_array_equal(b, n.b.detach().numpy().astype(np.float64))


def test_runtime_copies_match_originals():
    """Helpers duplicated to keep the runtime torch-free behave like the originals."""
    rng = np.random.default_rng(0)
    gd = np.concatenate([rng.normal(0, 1.5, 500), [0.0, 0.02, -0.02, 0.5, -0.5]])
    for t in [0.0, 0.02, 0.3]:
        np.testing.assert_array_equal(inference.gd_to_outcome(gd, t), fa.gd_to_outcome(gd, t))
    z = np.linspace(-6, 6, 1001)
    np.testing.assert_allclose(inference.normal_cdf(z), ndtr(z), atol=1e-12)
    for name, (cls, _) in fa.ACTIVATIONS.items():
        import torch
        zt = torch.linspace(-5, 5, 201, dtype=torch.float64)
        np.testing.assert_allclose(inference.ACTIVATIONS[name](zt.numpy()), cls()(zt).numpy(),
                                   atol=1e-12, err_msg=name)


FORBIDDEN = ["torch", "football_ann", "matplotlib", "sklearn", "scipy"]

_PROBE = """
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=90).run()
assert not at.exception, at.exception
at.toggle[0].set_value(True).run()
at.selectbox[0].set_value("Leicester").run()
assert not at.exception, at.exception
print(",".join(m for m in {forbidden!r} if m in sys.modules))
"""


def test_app_does_not_import_heavy_packages():
    """Run the whole app in a fresh interpreter and inspect sys.modules afterwards."""
    out = subprocess.run([sys.executable, "-c", _PROBE.format(forbidden=FORBIDDEN)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    assert loaded == "", f"app imported: {loaded}"
