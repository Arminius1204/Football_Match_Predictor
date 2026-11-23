# CLAUDE.md

## Project summary

Deep Learning (CSC403-8), CIA 1, Component I A. A PyTorch MLP predicts the final goal
difference (GD = home goals − away goals) of an English Premier League match from
information available **before kick-off**. Data: 11 seasons (2014-15 to 2024-25), 4,180
matches in `data/season-XXYY.csv` (football-data.co.uk format).

`football_ann.py` is the submitted assignment. It runs two experiments (E1: 5 activations,
E2: 6 architectures, 5 seeds each) and writes figures and metrics to `outputs/`. Best model
by validation MAE: 3 hidden layers [128, 64, 32], Sigmoid.

A second goal is a Streamlit web app (`app.py`) where the user picks a home and an away
team and gets a prediction. It is built **on top of** the assignment code and never
changes it.

## Hard rules

1. **Never modify the behaviour or results of `football_ann.py`.** It is the submitted
   assignment. No edits to its logic, constants, seeds, or outputs. `outputs/` and `data/`
   are read-only for app work. New code imports from it or sits beside it.
2. **No information leakage.** Features for a match use only matches finished before it.
   The scaler is fit on training data only. Hyperparameters and the draw threshold are
   chosen on validation only. The test set is touched once, for reporting.
3. **Fixed seeds.** Every training run calls `set_seed(seed)` (seeds 0-4 in the assignment).
   Reruns must reproduce the same numbers.
4. **Modular, commented code.** Small single-purpose functions, a docstring on each module
   and function, and comments that explain *why*. No copy-pasted logic. Reuse, don't duplicate.
5. **Reuse over rewrite.** If the app needs something `football_ann.py` already does,
   import it. If it can't be imported cleanly, say so before writing a second copy.
6. **No git, no attribution.** Run no git commands at all (not even status/diff) and add no
   AI attribution or co-author lines anywhere. The user handles version control. Report
   created/changed files in plain text instead.

## Feature pipeline (`football_ann.build_features`)

Matches are sorted by `(Date, HomeTeam)` and replayed in order. For each match, **before**
the result is applied:

- **Rolling form** per team: mean of the previous `WINDOW = 5` matches for `gf, ga, sf, sa,
  stf, sta, cf, pts` (goals for/against, shots for/against, shots on target for/against,
  corners for, points). The team needs `MIN_HISTORY = 3` earlier matches, otherwise the
  match is dropped. Form is carried across seasons.
- **Elo** per team: start 1500, K = 20, home advantage 60, goal-margin multiplier
  `log(|GD|+1) + 1`. Features are `elo_home`, `elo_away`, `elo_diff = elo_home + 60 − elo_away`.
- Only after the features are recorded are history and Elo updated with the result.

19 features: 3 Elo + 8 `home_*` form + 8 `away_*` form. Selection is by prefix
(`home_`, `away_`, `elo_`). Team names are never model inputs.

## Chronological split (no shuffling)

| Set | Seasons | Matches |
| --- | --- | --- |
| Train | 2014-15 to 2021-22 | 2,974 |
| Validation | 2022-23 | 377 |
| Test | 2023-24, 2024-25 | 754 |

`StandardScaler` is fit on train only. Early stopping (patience 25) uses validation loss.
The W/D/L draw band `|GD_hat| <= t` is tuned on validation only.

## Model and training

`MLP` is built from explicit `NeuronLayer` (`z = xWᵀ + b`) plus an activation, with a linear
output neuron. Training uses Adam (lr 1e-3, weight decay 1e-4), MSE loss, batch size 64,
up to 200 epochs.

## App architecture

Reuse `football_ann.py` via import. Do not copy it. Small modules in `src/` wrap it:

- `src/config.py`: paths, frozen split seasons, app model config, probability display rule.
- `src/features.py`: loads data via `fa.load_matches`. Team snapshots come from the
  "fixture trick" (append an unplayed fixture and call `fa.build_features`).
  `compose_features` builds pair features from two snapshots.
- `src/model.py`: split, train-only scaler, ensemble via `fa.train_model` (seeds 0-4),
  save/load.
- `src/evaluation.py`: MAE/RMSE/R², W/D/L accuracy, Normal(GD) probabilities, log loss,
  calibration.
- `src/inference.py`: NumPy-only runtime (feature composition, scaling, MLP forward pass,
  readouts). The training modules import their shared helpers from it.
- `src/predict.py`: `Predictor`, used by the app. It reads `weights.npz` plus the JSON
  artifacts and runs in NumPy.
- `src/charts.py`: Plotly figures.
- **Runtime boundary:** `app.py`, `src/config.py`, `src/predict.py`, `src/inference.py` and
  `src/charts.py` must never import torch, football_ann, matplotlib, scikit-learn or scipy.
  `tests/test_numpy_inference.py` enforces this and checks NumPy == PyTorch (< 1e-5) on all
  pairs.
- `requirements.txt` is the slim app runtime, which Streamlit Cloud reads.
  `requirements-train.txt` adds torch etc. for the assignment, training and tests.
- `train_app_model.py`: builds `artifacts/` offline and refuses to write if per-seed
  metrics don't match `results.json`. The app never trains.
- `app.py`: Streamlit UI only. No ML logic.

Model: [128, 64, 32], Sigmoid. Train 2014-22, val 2022-23, test 2023-25. Ensemble = mean of
seeds 0-4. Draw threshold and residual sigma are tuned on validation.

`data_live/` holds newer seasons (e.g. `season-2526.csv`). They are used ONLY to refresh
team Elo/form. They are never put in `data/`, because the assignment globs that folder.
Model, scaler, split and test set stay frozen.

Probabilities: GD ~ Normal(ensemble mean, validation residual std). They are shown only if
test log loss beats the training base-rate baseline AND every class has ECE <= 0.05
(`meta.json["probability_eval"]["show_probabilities"]`).

## Commands

```bash
pip install -r requirements-train.txt
python football_ann.py          # assignment pipeline (unchanged)
python train_app_model.py       # build artifacts/ for the app
streamlit run app.py            # launch the app
pytest tests/                   # leakage, reproducibility, parity checks
```

## Notes for Claude

- The assignment README lists one set of key results. Any app model must be checked against
  them (see the parity test) so the app does not quietly drift from the submitted work.
- Importing `football_ann` has light side effects (`os.makedirs("outputs")`,
  `torch.set_num_threads(4)`, `matplotlib.use("Agg")`). The data path is relative (`data`),
  so run from the repo root.
- Assignment data ends with 2024-25. App team state runs to the last match in
  `data_live/` (see `meta.json["state"]["ratings_as_of"]`).
- Verifying the assignment: run `football_ann.py` in a scratch copy and compare
  `results.json` ignoring `time_s` (wall-clock). Never rerun it in place.
- `pytest tests/` takes about 4 minutes (it retrains models and checks all 1,190 team pairs).
