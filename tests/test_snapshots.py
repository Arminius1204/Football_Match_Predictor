"""Team snapshots composed into pair features must equal the fixture trick, for EVERY pair."""

import json

import numpy as np

from src.features import compose_features, fixture_features, team_snapshots


def round_robin(teams):
    """Circle-method schedule: each round is a list of disjoint pairs, all pairs covered."""
    t = list(teams) + ([None] if len(teams) % 2 else [])
    n, rounds = len(t), []
    for _ in range(n - 1):
        rounds.append([(t[i], t[n - 1 - i]) for i in range(n // 2)
                       if t[i] is not None and t[n - 1 - i] is not None])
        t = [t[0]] + [t[-1]] + t[1:-1]
    return rounds


def test_snapshot_composition_matches_fixture_trick_for_every_pair(state_matches, cols):
    """For every ordered (home, away) pair, compose_features == build_features on that fixture.

    Fixtures are scored in disjoint batches (a round-robin, both orientations). A batch
    with no repeated team is equivalent to running each fixture alone, which the next
    test checks separately.
    """
    snaps, adv, _ = team_snapshots(state_matches)
    teams = sorted(snaps)
    checked = 0
    for rnd in round_robin(teams):
        for orient in (rnd, [(a, h) for h, a in rnd]):
            fx = fixture_features(state_matches, orient)
            assert len(fx) == len(orient)
            for (h, a), row in fx.iterrows():
                composed = compose_features(snaps[h], snaps[a], adv, cols)
                np.testing.assert_allclose(composed, row[cols].to_numpy(dtype=float),
                                           rtol=0, atol=1e-9, err_msg=f"{h} v {a}")
                checked += 1
    assert checked == len(teams) * (len(teams) - 1)


def test_batched_fixture_trick_equals_single_fixture(state_matches, cols):
    """Supports the batching argument: fixtures in a disjoint batch see untouched state."""
    batch = [("Arsenal", "Chelsea"), ("Sunderland", "Leeds"), ("Ipswich", "Burnley")]
    together = fixture_features(state_matches, batch)[cols]
    for pair in batch:
        alone = fixture_features(state_matches, [pair])[cols]
        np.testing.assert_array_equal(together.loc[[pair]].to_numpy(), alone.to_numpy())


def test_saved_snapshots_are_current(state_matches, artifacts_ready):
    """artifacts/team_snapshots.json equals a fresh replay of data/ + data_live/."""
    fresh, _, _ = team_snapshots(state_matches)
    saved = json.loads((artifacts_ready / "team_snapshots.json").read_text())
    assert saved.keys() == fresh.keys()
    for team in fresh:
        assert np.isclose(saved[team]["elo"], fresh[team]["elo"], atol=1e-9), team
        for k, v in fresh[team]["form"].items():
            assert np.isclose(saved[team]["form"][k], v, atol=1e-9), (team, k)
        assert saved[team]["last_match_date"] == fresh[team]["last_match_date"]
