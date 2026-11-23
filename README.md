# Football Match Goal-Difference Prediction with ANNs (PyTorch)

Deep Learning (CSC403-8) — CIA 1, Component I A — Akshat

Predicts the final goal difference (home − away) of English Premier League matches from
pre-match team form and Elo ratings, using a multilayer perceptron built from explicit
weight/bias neurons. Compares 5 activation functions (E1) and 6 network configurations (E2),
each over 5 random seeds.

## Run

```bash
pip install -r requirements-train.txt
python football_ann.py        # ~2 minutes on a laptop CPU
```

## Contents

| Path | What it is |
| --- | --- |
| `football_ann.py` | Full pipeline: data loading, leak-free features, model, gradient check, training, experiments, figures |
| `data/season-XXYY.csv` | EPL 2014-15 to 2024-25 (football-data.co.uk format, via github.com/datasets/football-datasets) |
| `outputs/results.json` | All metrics, per-configuration mean ± std, gradient-check results |
| `outputs/table_E1_activations.csv`, `outputs/table_E2_architectures.csv` | Result tables |
| `outputs/fig1`–`fig6` | Figures used in the report |
| `app.py`, `src/`, `artifacts/` | Streamlit web app, its support code and trained model files (see below) |

## Key results (test set: 2023-24 + 2024-25, 754 matches)

| Model | Test MAE (goals) | Test R² |
| --- | --- | --- |
| Mean baseline | 1.531 | −0.002 |
| Single linear neuron | 1.359 | 0.217 |
| MLP [16], Sigmoid | 1.350 | 0.218 |
| MLP [64, 32], Sigmoid | 1.353 | 0.214 |
| MLP [64, 32], ReLU | 1.371 | 0.190 |

Seeds are fixed (0–4), so rerunning reproduces these numbers exactly.

## Web app

`app.py` is a Streamlit app: pick a home and an away club and it predicts the goal difference.
It serves the assignment's best configuration ([128, 64, 32], Sigmoid) as an average of the five
seed models. A gauge shows the margin, the spread across seeds and a wording band (too close
to call / slight edge / clear edge), next to each club's Elo and last-five form.
`football_ann.py` and its results are unchanged. The app imports nothing from it at runtime.

```bash
pip install -r requirements-train.txt
python train_app_model.py     # rebuild artifacts/ (~1 minute)
pytest tests/                 # leakage, parity and NumPy-vs-PyTorch checks (~4 minutes)
streamlit run app.py          # needs only requirements.txt (no PyTorch)
```

`train_app_model.py` refuses to write artifacts unless the retrained models reproduce
`outputs/results.json`. The app runs the ensemble in NumPy from `artifacts/weights.npz`.

Model caveats:

- **Frozen model.** Training, scaling and the test results are the assignment's: train 2014-15
  to 2021-22, validation 2022-23, test 2023-24 + 2024-25. On the 754 test matches the ensemble
  scores MAE 1.355 goals and R² 0.209, against 1.531 and −0.002 for always predicting the
  average margin.
- **Ratings as of 24 May 2026.** `data_live/season-2526.csv` refreshes team Elo and form only.
  It never touches the model, and it is kept out of `data/` so it can't change the
  assignment's results. Transfers, injuries and 2026-27 promotions are not reflected. Clubs
  outside the Premier League keep a frozen, likely stale rating.
- **No percentages.** Win/draw/loss probabilities from a Normal model of the predicted margin
  beat a base-rate baseline on log loss (0.966 vs 1.068), but did not clear a calibration bar
  set in advance (home-win calibration error 0.056 vs a limit of 0.05). Home wins were overstated
  by about 5 points on average (48.0% predicted vs 43.4% observed), likely because home advantage
  weakened in 2023-25. The app shows the margin instead. The wording bands were fixed on 2022-23.
  On the test set the favourite won 42% of "too close to call" matches and 69% of "clear edge"
  matches.
