"""Pre-match features for any (home, away) pair, built on football_ann.build_features.

The Elo and rolling-form logic lives in exactly one place: football_ann.build_features.
This module never re-implements it. It uses two ideas.

1. **Fixture trick.** build_features records a match's features BEFORE applying its
   result. So if we append a not-yet-played fixture after the last real match, with a
   placeholder scoreline, the row it returns holds the true pre-match features, and the
   placeholder can't affect them. tests/test_no_leakage.py checks this.

2. **Disjoint batching.** A fixture only updates the state of its own two teams. If no
   team appears twice in a batch, every fixture in the batch sees exactly the state left
   by the real matches, as if it had been run alone. One build_features call (~1 s) can
   therefore score many fixtures at once.

Team snapshots (current Elo + last-5 form means) come from one batched call. Pair
features are then composed from two snapshots at prediction time (compose_features),
and tests/test_snapshots.py checks that this equals the fixture trick for every pair.
"""

import contextlib
import glob
import os

import numpy as np
import pandas as pd

import football_ann as fa
from src import config
# Pair composition lives in the NumPy-only runtime module (the app uses it without torch).
from src.inference import compose_features  # noqa: F401  (re-exported for callers/tests)

FIXTURE_SEASON = "fixture"  # Season label marking appended, unplayed fixtures


# ---------------------------------------------------------------------------
# Data loading (delegates to football_ann.load_matches)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _data_dir(path):
    """Point football_ann.load_matches at another folder for the duration of a call.

    This changes a module attribute at runtime only, never the file. It also makes
    loading independent of the current working directory.
    """
    previous = fa.DATA_DIR
    fa.DATA_DIR = str(path)
    try:
        yield
    finally:
        fa.DATA_DIR = previous


def _load_dir(path):
    """Load every season-*.csv in ``path`` with the assignment's own loader."""
    with _data_dir(path):
        return fa.load_matches()


def load_assignment_matches():
    """The exact matches the assignment trains and tests on (data/ only)."""
    return _load_dir(config.DATA_DIR)


def live_season_files():
    """Seasons in data_live/ (used only to refresh team state)."""
    return sorted(glob.glob(os.path.join(config.LIVE_DATA_DIR, "season-*.csv")))


def load_state_matches(include_live=True):
    """All matches used to compute CURRENT team state: data/ plus, optionally, data_live/.

    The model never sees this frame. It only feeds the Elo/form replay that gives each
    team's latest snapshot.
    """
    df = load_assignment_matches()
    if include_live and live_season_files():
        live = _load_dir(config.LIVE_DATA_DIR)
        overlap = set(df.Season) & set(live.Season)
        assert not overlap, f"season(s) {overlap} present in both data/ and data_live/"
        df = pd.concat([df, live], ignore_index=True)
        # Same ordering rule as football_ann.load_matches, so the replay is identical.
        df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
def feature_columns(feats):
    """Model input columns, chosen exactly as in football_ann.main() (by prefix)."""
    return [c for c in feats.columns
            if c.startswith(("home_", "away_", "elo_")) and c not in ("HomeTeam", "AwayTeam")]


def form_keys(feature_cols):
    """Rolling-form statistic names (gf, ga, ...), read from the feature columns."""
    return [c[len("home_"):] for c in feature_cols if c.startswith("home_")]


# ---------------------------------------------------------------------------
# Fixture trick
# ---------------------------------------------------------------------------
def _fixture_rows(df, pairs, placeholder):
    """Unplayed fixtures dated one day after the last match, with a placeholder score."""
    date = df["Date"].max() + pd.Timedelta(days=1)
    hg, ag = placeholder
    rows = []
    for home, away in pairs:
        row = {c: np.nan for c in df.columns}
        row.update(Date=date, Season=FIXTURE_SEASON, HomeTeam=home, AwayTeam=away,
                   FTHG=hg, FTAG=ag, GD=hg - ag, FTR="H" if hg > ag else ("A" if ag > hg else "D"),
                   HS=0, AS=0, HST=0, AST=0, HC=0, AC=0)
        rows.append(row)
    return pd.DataFrame(rows, columns=df.columns)


def fixture_features(df, pairs, placeholder=(0, 0)):
    """Pre-match features for unplayed fixtures, computed by football_ann.build_features.

    ``pairs`` is a list of (home, away). No team may appear twice in one call, since a
    later fixture would then see an earlier fixture's placeholder result. Returns a
    DataFrame indexed by (HomeTeam, AwayTeam). A pair is missing if a team has fewer
    than MIN_HISTORY matches.
    """
    teams = [t for p in pairs for t in p]
    if len(teams) != len(set(teams)):
        raise ValueError("fixtures in one batch must involve disjoint teams")
    full = pd.concat([df, _fixture_rows(df, pairs, placeholder)], ignore_index=True)
    feats = fa.build_features(full)
    out = feats[feats.Season == FIXTURE_SEASON]
    return out.set_index(["HomeTeam", "AwayTeam"])


def home_advantage(feats):
    """Recover build_features' Elo home advantage from its output.

    The constant is a local variable inside build_features, so it can't be imported.
    Every row satisfies elo_diff = elo_home + HOME_ADV - elo_away, so we read it off the
    data and check that it is constant.
    """
    adv = (feats["elo_diff"] - feats["elo_home"] + feats["elo_away"]).to_numpy()
    assert np.allclose(adv, adv[0]), "home advantage is not constant"
    return float(np.round(adv[0], 9))


# ---------------------------------------------------------------------------
# Team snapshots and pair composition
# ---------------------------------------------------------------------------
def _disjoint_batches(teams):
    """Split teams into fixture batches where no team repeats within a batch.

    Batch 1 pairs teams up: (t0, t1), (t2, t3), ... With an odd count, the last team
    gets a second batch of its own against t0.
    """
    batches = [[(teams[i], teams[i + 1]) for i in range(0, len(teams) - 1, 2)]]
    if len(teams) % 2 == 1 and len(teams) > 1:
        batches.append([(teams[-1], teams[0])])
    return batches


def _recent_matches(df, team, n):
    """The team's last ``n`` real matches, for display (raw results, not features)."""
    m = df[(df.HomeTeam == team) | (df.AwayTeam == team)].tail(n)
    out = []
    for _, r in m.iterrows():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        out.append({"date": r.Date.strftime("%Y-%m-%d"), "season": r.Season,
                    "opponent": r.AwayTeam if home else r.HomeTeam, "venue": "H" if home else "A",
                    "gf": int(gf), "ga": int(ga),
                    "result": "W" if gf > ga else ("D" if gf == ga else "L")})
    return out


def _elo_history(feats, team):
    """Pre-match Elo before each of the team's matches that build_features kept."""
    h = feats.loc[feats.HomeTeam == team, ["Date", "elo_home"]].rename(columns={"elo_home": "elo"})
    a = feats.loc[feats.AwayTeam == team, ["Date", "elo_away"]].rename(columns={"elo_away": "elo"})
    hist = pd.concat([h, a]).sort_values("Date")
    return [{"date": d.strftime("%Y-%m-%d"), "elo": float(e)} for d, e in zip(hist.Date, hist.elo)]


def team_snapshots(df):
    """Current state of every team after the last match in ``df``.

    Returns (snapshots, home_adv, feature_cols). ``snapshots`` maps team -> dict with:
      elo, form (last-5 means used as model inputs), last_match_date, last_season,
      seasons, n_matches, recent (last-5 raw results) and elo_history (pre-match Elo
      over time, ending with the current rating).
    """
    feats = fa.build_features(df)                       # all real matches
    cols = feature_columns(feats)
    keys = form_keys(cols)
    adv = home_advantage(feats)
    window = fa.WINDOW

    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    state = {}
    for batch in _disjoint_batches(teams):
        fx = fixture_features(df, batch)
        for (home, away), r in fx.iterrows():
            # A team's form is from its own perspective (gf = goals it scored),
            # so home_* / away_* describe the team the same way whichever side it plays.
            state.setdefault(home, {"elo": float(r.elo_home),
                                    "form": {k: float(r[f"home_{k}"]) for k in keys}})
            state.setdefault(away, {"elo": float(r.elo_away),
                                    "form": {k: float(r[f"away_{k}"]) for k in keys}})

    snapshots = {}
    for team in teams:
        if team not in state:          # fewer than MIN_HISTORY matches: cannot be scored
            continue
        played = df[(df.HomeTeam == team) | (df.AwayTeam == team)]
        last_date = played.Date.max()
        history = _elo_history(feats, team)
        # Add the current rating as the final point (just after the last match).
        history.append({"date": (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        "elo": state[team]["elo"]})
        snapshots[team] = {
            **state[team],
            "last_match_date": last_date.strftime("%Y-%m-%d"),
            "last_season": str(played.Season.iloc[-1]),
            "seasons": sorted(set(played.Season)),
            "n_matches": int(len(played)),
            "recent": _recent_matches(df, team, window),
            "elo_history": history,
        }
    return snapshots, adv, cols


