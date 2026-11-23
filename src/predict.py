"""Inference for the app: team snapshots -> features -> scale -> ensemble -> GD (+ W/D/L).

Torch-free by design. It reads only the files written by train_app_model.py
(weights.npz, scaler.json, meta.json, team_snapshots.json) and runs the forward pass in
NumPy (src/inference.py). Runtime dependencies are numpy and the standard library.
No training, no data replay. The app imports Predictor and nothing else from the ML stack.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from src import config
from src.inference import (assign_band, compose_features, ensemble_members, gd_to_outcome,
                           lean_side, load_numpy_ensemble, outcome_probabilities, transform)

OUTCOMES = ["H", "D", "A"]


@dataclass
class Prediction:
    """Everything the UI needs for one fixture."""
    home: str
    away: str
    gd: float                          # ensemble mean predicted goal difference
    gd_members: list                   # one prediction per seed (spread = model disagreement)
    outcome: str                       # 'H' / 'D' / 'A' from the validation-tuned draw band
    probs: dict | None                 # {'H','D','A'} model estimate, or None if not validated
    band: str | None                   # wording band, or None if bands failed their test
    lean: str                          # 'H' or 'A': side the model leans toward
    features: dict                     # raw (unscaled) inputs, for transparency
    warnings: list = field(default_factory=list)


class Predictor:
    """Loads artifacts once. Use predict(home, away) for each fixture."""

    def __init__(self, artifact_dir=config.ARTIFACT_DIR):
        d = Path(artifact_dir)
        self.members, self.activation = load_numpy_ensemble(d / "weights.npz")
        scaler = json.loads((d / "scaler.json").read_text())
        self.feature_cols = scaler["feature_cols"]
        self.mean, self.scale = np.array(scaler["mean"]), np.array(scaler["scale"])
        self.meta = json.loads((d / "meta.json").read_text())
        self.snapshots = json.loads((d / "team_snapshots.json").read_text())
        self.home_adv = self.meta["home_adv"]
        self.threshold = self.meta["draw_threshold"]
        self.sigma = self.meta["sigma"]
        self.show_probs = self.meta["probability_eval"]["show_probabilities"]
        self.show_bands = self.meta["bands"]["show_bands"]
        self.band_cutoffs = self.meta["bands"]["cutoffs"]
        self.band_names = self.meta["bands"]["names"]

    # ---- team lists --------------------------------------------------------
    @property
    def latest_season(self):
        """Most recent season in the team-state data (e.g. '2526')."""
        return self.meta["state"]["latest_season"]

    def teams(self, include_all=False):
        """Default: clubs in the latest state season. include_all: every scorable team."""
        if include_all:
            return sorted(self.snapshots)
        return sorted(t for t, s in self.snapshots.items() if s["last_season"] == self.latest_season)

    def is_current(self, team):
        """True if the team played in the latest season, so its rating is fresh."""
        return self.snapshots[team]["last_season"] == self.latest_season

    def team_warnings(self, team):
        """Staleness note for clubs whose Elo/form stopped updating (no lower-league data)."""
        s = self.snapshots[team]
        if self.is_current(team):
            return []
        last = date.fromisoformat(s["last_match_date"])
        season = f"20{s['last_season'][:2]}-{s['last_season'][2:]}"
        return [f"{team} has not played in the Premier League since {last.day} {last:%B %Y} "
                f"(the {season} season). Its rating is frozen at that point and is likely stale."]

    # ---- prediction --------------------------------------------------------
    def features_for(self, home, away):
        """Unscaled feature vector for the fixture, composed from the two snapshots."""
        return compose_features(self.snapshots[home], self.snapshots[away],
                                self.home_adv, self.feature_cols)

    def member_predictions(self, X_raw):
        """Per-seed predicted GD for raw feature rows: shape (n_models, n_rows)."""
        return ensemble_members(self.members, self.activation,
                                transform(np.atleast_2d(X_raw), self.mean, self.scale))

    def predict(self, home, away):
        """Predict goal difference (home - away) for an unplayed fixture."""
        if home == away:
            raise ValueError("home and away teams must differ")
        x = self.features_for(home, away)
        members = self.member_predictions(x)[:, 0]
        gd = float(members.mean())
        outcome = str(gd_to_outcome(np.array([gd]), self.threshold)[0])
        probs = None
        if self.show_probs:
            probs = dict(zip(OUTCOMES, map(float, outcome_probabilities(gd, self.sigma)[0])))
        band = None
        if self.show_bands:
            band = self.band_names[int(assign_band([gd], self.band_cutoffs)[0])]
        return Prediction(home=home, away=away, gd=gd, gd_members=members.tolist(),
                          outcome=outcome, probs=probs, band=band, lean=str(lean_side(gd)),
                          features=dict(zip(self.feature_cols, map(float, x))),
                          warnings=self.team_warnings(home) + self.team_warnings(away))
