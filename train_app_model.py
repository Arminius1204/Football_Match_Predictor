"""Build artifacts/ for the Streamlit app (run once, offline; the app never trains).

    python train_app_model.py

Steps
  1. Load the assignment's data (data/ only) and features via football_ann.
  2. Chronological split + train-only scaler, as in football_ann.main().
  3. Train the assignment's best config ([128, 64, 32], Sigmoid) for seeds 0-4.
  4. Check parity: per-seed test metrics must match outputs/results.json.
  5. Ensemble = mean of the 5 models. Tune the draw band and the residual sigma on
     VALIDATION, then score once on TEST (2023-25): MAE/RMSE/R², W/D/L accuracy,
     log loss vs base rate, calibration.
  6. Refresh team state (Elo + form) from data/ + data_live/. This touches team
     snapshots only: model, scaler, split and test set stay frozen.
  7. Write artifacts/: model.pt (PyTorch), weights.npz (same weights for the torch-free
     app), scaler.json, meta.json, team_snapshots.json.
"""

import json
import os

import numpy as np
import torch

import football_ann as fa
from src import config, evaluation as ev
from src.features import (feature_columns, live_season_files, load_assignment_matches,
                          load_state_matches, team_snapshots)
from src.model import (export_numpy_weights, predict_ensemble, prepare_data, save_models,
                       save_scaler, train_ensemble)


def check_parity(per_seed, epochs):
    """Compare per-seed metrics with the assignment's results.json entry for this config.

    The assignment reports mean ± std over seeds 0-4. Matching both for every metric
    shows the app's models are the assignment's models.
    """
    ref = json.loads(config.ASSIGNMENT_RESULTS.read_text())
    entry = next(e for e in ref["E2"] if e["name"] == config.APP_CONFIG_NAME)
    report, ok = {}, True
    for k in ["val_MAE", "MAE", "RMSE", "R2", "WDL_acc"]:
        vals = [m[k] for m in per_seed]
        mine = (float(np.mean(vals)), float(np.std(vals)))
        theirs = tuple(entry["metrics"][k])
        match = bool(np.allclose(mine, theirs, atol=1e-6))
        ok &= match
        report[k] = {"ours": mine, "results_json": theirs, "match": match}
    ep = (float(np.mean(epochs)), float(np.std(epochs)))
    report["best_epoch"] = {"ours": ep, "results_json": tuple(entry["metrics"]["best_epoch"]),
                            "match": bool(np.allclose(ep, entry["metrics"]["best_epoch"]))}
    ok &= report["best_epoch"]["match"]
    return ok, report


def main():
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)

    # 1-2. Frozen assignment data, features, split, scaler
    df = load_assignment_matches()
    feats = fa.build_features(df)
    cols = feature_columns(feats)
    data, scaler, (tr, va, te) = prepare_data(feats, cols)
    print(f"Assignment data: {len(df)} matches, {len(feats)} usable, {len(cols)} features")
    print(f"Split  train {len(tr)} | val {len(va)} | test {len(te)}")

    # 3. Train the ensemble members exactly as the assignment does
    models, epochs = train_ensemble(data)
    per_seed = [fa.evaluate(m, data)[0] for m in models]
    for s, m, ep in zip(config.SEEDS, per_seed, epochs):
        print(f"  seed {s}: test MAE {m['MAE']:.4f}  RMSE {m['RMSE']:.4f}  R2 {m['R2']:.4f}  "
              f"WDL {m['WDL_acc']:.4f}  best epoch {ep}")

    # 4. Parity with the submitted results
    parity_ok, parity = check_parity(per_seed, epochs)
    print("Parity with outputs/results.json:", "OK" if parity_ok else "MISMATCH")
    if not parity_ok:
        print(json.dumps(parity, indent=2))
        raise SystemExit("Refusing to write artifacts: models differ from the assignment's.")

    # 5. Ensemble: tune on validation, score once on test
    yva, yte = data["yva"].numpy(), data["yte"].numpy()
    pva, pte = predict_ensemble(models, data["Xva"]), predict_ensemble(models, data["Xte"])
    threshold = fa.best_threshold(pva, data["ftr_va"])            # validation only
    sigma = ev.residual_sigma(yva, pva)                            # validation only
    ens_test = {**ev.regression_metrics(yte, pte),
                "WDL_acc": ev.wdl_accuracy(pte, data["ftr_te"], threshold)}
    ens_val = ev.regression_metrics(yva, pva)
    mean_base = ev.regression_metrics(yte, np.full_like(yte, data["ytr"].mean().item()))

    baseline_probs = ev.base_rates(tr.FTR.values)                  # training base rates
    probs_te = ev.outcome_probabilities(pte, sigma)
    prob_eval = ev.probability_report(probs_te, data["ftr_te"], baseline_probs,
                                      config.MAX_ECE_TO_SHOW)

    # Wording bands: cutoffs from validation |GD_hat|, judged once on test
    cutoffs = ev.band_cutoffs(pva, config.BAND_QUANTILES)          # validation only
    bands = ev.band_report(pte, data["ftr_te"], cutoffs, config.BAND_NAMES,
                           config.BAND_MIN_GAP, config.BAND_MAX_P)
    bands["names"] = config.BAND_NAMES
    bands["val_check"] = ev.band_report(pva, data["ftr_va"], cutoffs, config.BAND_NAMES,
                                        config.BAND_MIN_GAP, config.BAND_MAX_P)["rows"]

    # 6. Team state: replay data/ + data_live/ for current Elo and form
    state_df = load_state_matches(include_live=True)
    snapshots, home_adv, state_cols = team_snapshots(state_df)
    assert state_cols == cols, "feature columns differ between model and team state"
    latest = str(state_df.Season.iloc[-1])

    # 7. Write artifacts
    save_models(models, len(cols), config.APP_HIDDEN, config.APP_ACTIVATION, config.SEEDS,
                config.ARTIFACT_DIR / "model.pt")
    save_scaler(scaler, cols, config.ARTIFACT_DIR / "scaler.json")
    # Same weights as model.pt, as plain arrays for the torch-free app runtime.
    export_numpy_weights(models, config.APP_ACTIVATION, config.ARTIFACT_DIR / "weights.npz")
    meta = {
        "model": {"hidden": config.APP_HIDDEN, "activation": config.APP_ACTIVATION,
                  "seeds": config.SEEDS, "best_epochs": epochs, "ensemble": "mean of members",
                  "torch_version": torch.__version__},
        "split": {"train_seasons": config.TRAIN_SEASONS, "val_seasons": config.VAL_SEASONS,
                  "test_seasons": config.TEST_SEASONS,
                  "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                  "test_period": [te.Date.min().strftime("%Y-%m-%d"), te.Date.max().strftime("%Y-%m-%d")]},
        "feature_cols": cols,
        "home_adv": home_adv,
        "draw_threshold": threshold,
        "sigma": sigma,
        "ensemble_test_metrics": ens_test,
        "ensemble_val_metrics": ens_val,
        "per_seed_test_metrics": per_seed,
        "baseline_test_metrics": {**mean_base,
                                  "WDL_acc_always_home": float((data["ftr_te"] == "H").mean())},
        "parity_with_results_json": parity,
        "probability_eval": prob_eval,
        "bands": bands,
        "state": {"ratings_as_of": state_df.Date.max().strftime("%Y-%m-%d"),
                  "latest_season": latest,
                  "live_files": [os.path.basename(p) for p in live_season_files()],
                  "n_matches": len(state_df), "seasons": sorted(set(state_df.Season))},
    }
    (config.ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    (config.ARTIFACT_DIR / "team_snapshots.json").write_text(json.dumps(snapshots, indent=2))

    # Summary
    print(f"\nEnsemble (5 seeds) on TEST {meta['split']['test_period']}:")
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in ens_test.items()))
    print("Baselines: " + "  ".join(f"{k} {v:.4f}" for k, v in meta["baseline_test_metrics"].items()))
    print(f"Draw band |GD| <= {threshold:.2f}   sigma (val residual std) {sigma:.4f}")
    print(f"Log loss  model {prob_eval['log_loss_model']:.4f}  vs base-rate {prob_eval['log_loss_baseline']:.4f}")
    print(f"Brier     model {prob_eval['brier_model']:.4f}  vs base-rate {prob_eval['brier_baseline']:.4f}")
    print("ECE      ", {k: round(v, 4) for k, v in prob_eval["ece"].items()})
    for o, rows in prob_eval["calibration"].items():
        print(f"  Calibration {o}: " + " | ".join(
            f"{r['bin']} n={r['n']} pred {r['mean_pred']:.2f} obs {r['observed']:.2f}" for r in rows))
    print("Show probabilities:", prob_eval["show_probabilities"], "-", prob_eval["reason"])
    print(f"Wording bands, cutoffs on validation |GD_hat|: {[round(c, 3) for c in cutoffs]}")
    for r in bands["rows"]:
        print(f"  TEST {r['band']:<18} n={r['n']:<4} lean won {r['lean_won']:.3f}  draw {r['draw']:.3f}  "
              f"lean lost {r['lean_lost']:.3f}")
    print("Show bands:", bands["show_bands"], "-", bands["reason"])
    print(f"Team state: {len(snapshots)} teams, ratings as of {meta['state']['ratings_as_of']} "
          f"(latest season {latest}, live files {meta['state']['live_files']})")
    print(f"Artifacts written to {config.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
