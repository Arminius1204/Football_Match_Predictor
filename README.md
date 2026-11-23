# Football Match Goal-Difference Prediction with ANNs (PyTorch)

Deep Learning (CSC403-8) — CIA 1, Component I A — Akshat

Predicts the final goal difference (home − away) of English Premier League matches from
pre-match team form and Elo ratings, using a multilayer perceptron built from explicit
weight/bias neurons. Compares 5 activation functions (E1) and 6 network configurations (E2),
each over 5 random seeds.

## Run

```bash
pip install -r requirements.txt
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

## Key results (test set: 2023-24 + 2024-25, 754 matches)

| Model | Test MAE (goals) | Test R² |
| --- | --- | --- |
| Mean baseline | 1.531 | −0.002 |
| Single linear neuron | 1.359 | 0.217 |
| MLP [16], Sigmoid | 1.350 | 0.218 |
| MLP [64, 32], Sigmoid | 1.353 | 0.214 |
| MLP [64, 32], ReLU | 1.371 | 0.190 |

Seeds are fixed (0–4), so rerunning reproduces these numbers exactly.
