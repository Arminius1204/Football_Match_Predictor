"""Paths and frozen settings shared by the training script, the app and the tests.

Imported by the deployed app, so it must stay dependency-free (no football_ann/torch).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Assignment data (read by football_ann.load_matches). Never add files here: the
# assignment globs data/season-*.csv, so an extra season would change its results.
DATA_DIR = REPO_ROOT / "data"

# Newer seasons used ONLY to refresh team state (Elo + rolling form) for the app.
# They never reach the model, the scaler, the split or the test set.
LIVE_DATA_DIR = REPO_ROOT / "data_live"

ARTIFACT_DIR = REPO_ROOT / "artifacts"
ASSIGNMENT_RESULTS = REPO_ROOT / "outputs" / "results.json"

# Chronological split. These lists are local variables inside football_ann.main(),
# so they cannot be imported. tests/test_parity.py checks that they give the same
# split sizes as outputs/results.json.
TRAIN_SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122"]
VAL_SEASONS = ["2223"]
TEST_SEASONS = ["2324", "2425"]

# App model: the assignment's best configuration by validation MAE
# (results.json -> best_overall), served as an ensemble over the assignment's seeds.
APP_HIDDEN = [128, 64, 32]
APP_ACTIVATION = "Sigmoid"
APP_CONFIG_NAME = "3 hidden [128, 64, 32]"  # its name in results.json["E2"]
# The assignment's seeds. Written out (not imported from football_ann) so the app can load
# this module without torch; tests/test_parity.py asserts it equals football_ann.SEEDS.
SEEDS = [0, 1, 2, 3, 4]

# Probability display rule. It was fixed before any probabilities were evaluated:
# show W/D/L percentages only if they beat the base-rate baseline on test log loss
# AND every class's expected calibration error is at most this value.
MAX_ECE_TO_SHOW = 0.05

# Wording bands for the plain readout, fixed before any band was evaluated.
# Cutoffs are the terciles of |predicted GD| on the 2022-23 VALIDATION season (they use
# prediction sizes only, never outcomes). Bands, from smallest |GD_hat| to largest:
BAND_NAMES = ["Too close to call", "Slight edge", "Clear edge"]
BAND_QUANTILES = [1 / 3, 2 / 3]
# Keep the bands in the app only if, on TEST, the leaned-to side wins at least this much
# more often in "Clear edge" than in "Too close to call", AND a one-sided two-proportion
# z-test on that gap gives p < BAND_MAX_P. Otherwise show GD + seed spread only.
BAND_MIN_GAP = 0.10
BAND_MAX_P = 0.05
