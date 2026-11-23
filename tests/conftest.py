"""Shared fixtures. Expensive objects (feature replay, ensemble training) are built once per session."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # football_ann uses relative paths (data/, outputs/)

import football_ann as fa  # noqa: E402
from src import config  # noqa: E402
from src.features import feature_columns, load_assignment_matches, load_state_matches  # noqa: E402
from src.model import prepare_data, train_ensemble  # noqa: E402


@pytest.fixture(scope="session")
def matches():
    """Assignment matches (data/ only)."""
    return load_assignment_matches()


@pytest.fixture(scope="session")
def feats(matches):
    return fa.build_features(matches)


@pytest.fixture(scope="session")
def cols(feats):
    return feature_columns(feats)


@pytest.fixture(scope="session")
def prepared(feats, cols):
    """(data, scaler, (tr, va, te)) exactly as the training script builds them."""
    return prepare_data(feats, cols)


@pytest.fixture(scope="session")
def ensemble(prepared):
    """Freshly trained app ensemble (seeds 0-4): (models, best_epochs). ~20 s."""
    return train_ensemble(prepared[0])


@pytest.fixture(scope="session")
def state_matches():
    """data/ + data_live/: the frame team snapshots are computed from."""
    return load_state_matches(include_live=True)


@pytest.fixture(scope="session")
def artifacts_ready():
    needed = ["model.pt", "scaler.json", "meta.json", "team_snapshots.json"]
    if not all((config.ARTIFACT_DIR / f).exists() for f in needed):
        pytest.skip("artifacts/ missing: run `python train_app_model.py` first")
    return config.ARTIFACT_DIR


@pytest.fixture(scope="session")
def predictor(artifacts_ready):
    from src.predict import Predictor
    return Predictor(artifacts_ready)
