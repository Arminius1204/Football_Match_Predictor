"""Information-leakage checks: features, scaler, split and the frozen model vs live data."""

import glob

import numpy as np
import pandas as pd

import football_ann as fa
from src import config
from src.features import fixture_features


def test_placeholder_score_does_not_affect_fixture_features(state_matches, cols):
    """The fixture trick is leak-free: the placeholder result can't change the features."""
    pairs = [("Arsenal", "Chelsea"), ("Liverpool", "Man City"), ("Everton", "Fulham")]
    a = fixture_features(state_matches, pairs, placeholder=(0, 0))[cols]
    b = fixture_features(state_matches, pairs, placeholder=(9, 0))[cols]
    c = fixture_features(state_matches, pairs, placeholder=(0, 7))[cols]
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


def test_features_use_only_past_matches(matches, feats, cols):
    """Truncating the data right after a match leaves that match's features unchanged.

    If build_features peeked at later matches, the truncated replay would differ.
    """
    for i in [400, 1500, 2900, 3400, len(matches) - 1]:
        m = matches.iloc[i]
        full = feats[(feats.Date == m.Date) & (feats.HomeTeam == m.HomeTeam)]
        trunc = fa.build_features(matches.iloc[: i + 1]).iloc[-1]
        assert len(full) == 1 and trunc.HomeTeam == m.HomeTeam
        np.testing.assert_array_equal(full[cols].to_numpy()[0], trunc[cols].to_numpy(dtype=float))


def test_scaler_fit_on_training_data_only(prepared, cols):
    _, scaler, (tr, va, te) = prepared
    np.testing.assert_allclose(scaler.mean_, tr[cols].mean().to_numpy(), rtol=0, atol=1e-10)
    all_mean = pd.concat([tr, va, te])[cols].mean().to_numpy()
    assert not np.allclose(scaler.mean_, all_mean), "scaler looks fit on more than train"


def test_split_is_chronological_and_disjoint(prepared):
    _, _, (tr, va, te) = prepared
    assert tr.Date.max() < va.Date.min() and va.Date.max() < te.Date.min()
    assert set(tr.Season).isdisjoint(va.Season) and set(va.Season).isdisjoint(te.Season)


def test_live_data_never_reaches_the_model(matches, artifacts_ready):
    """data_live/ seasons only refresh team state: none is in data/ or the frozen split."""
    import json
    live_seasons = {p[-8:-4] for p in glob.glob(str(config.LIVE_DATA_DIR / "season-*.csv"))}
    assert not glob.glob(str(config.DATA_DIR / "season-25*.csv")), "newer season leaked into data/"
    assert live_seasons.isdisjoint(set(matches.Season))
    split = config.TRAIN_SEASONS + config.VAL_SEASONS + config.TEST_SEASONS
    assert live_seasons.isdisjoint(split)
    meta = json.loads((artifacts_ready / "meta.json").read_text())
    assert meta["split"]["test_period"][1] <= "2025-05-25"
    assert (meta["split"]["n_train"], meta["split"]["n_val"], meta["split"]["n_test"]) == (2974, 377, 754)


def test_tuning_quantities_come_from_validation(prepared, ensemble, artifacts_ready):
    """Draw threshold and sigma in meta.json are reproducible from VALIDATION data alone."""
    import json
    from src import evaluation as ev
    from src.model import predict_ensemble
    data = prepared[0]
    pva = predict_ensemble(ensemble[0], data["Xva"])
    meta = json.loads((artifacts_ready / "meta.json").read_text())
    assert meta["draw_threshold"] == fa.best_threshold(pva, data["ftr_va"])
    assert np.isclose(meta["sigma"], ev.residual_sigma(data["yva"].numpy(), pva), atol=1e-12)
    np.testing.assert_allclose(meta["bands"]["cutoffs"],
                               ev.band_cutoffs(pva, config.BAND_QUANTILES), atol=1e-12)
