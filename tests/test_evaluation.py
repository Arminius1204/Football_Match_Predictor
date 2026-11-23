"""Unit checks for the probability maths (fast, no training)."""

import numpy as np

from src import evaluation as ev


def test_probabilities_sum_to_one_and_follow_definition():
    mu = np.array([-2.0, 0.0, 0.3, 2.5])
    p = ev.outcome_probabilities(mu, sigma=1.7)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)
    # mu = 0 is symmetric: P(home) == P(away)
    assert np.isclose(p[1, 0], p[1, 2])
    # Higher mu -> more likely home win
    assert np.all(np.diff(p[:, 0]) > 0) and np.all(np.diff(p[:, 2]) < 0)


def test_log_loss_of_base_rates_equals_entropy_on_matching_data():
    ftr = np.array(["H"] * 5 + ["D"] * 3 + ["A"] * 2)
    rates = ev.base_rates(ftr)
    probs = np.tile(rates, (len(ftr), 1))
    entropy = -np.sum(rates * np.log(rates))
    assert np.isclose(ev.log_loss(probs, ftr), entropy)


def test_calibration_table_perfect_case():
    probs = np.tile([0.5, 0.25, 0.25], (8, 1))
    ftr = np.array(["H", "H", "H", "H", "D", "D", "A", "A"])
    table = ev.calibration_table(probs, ftr)
    ece = ev.expected_calibration_error(table, len(ftr))
    assert all(np.isclose(v, 0) for v in ece.values())
