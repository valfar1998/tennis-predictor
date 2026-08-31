"""Match predictor: blend Elo + Markov + ML."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.data_update.altitude import altitude_serve_boost, lookup_altitude
from modules.data_update.cpi import cpi_serve_adjustment, lookup_cpi
from modules.data_update.tennisratio import lookup_player_skills, skill_serve_adjustment
from modules.data_update.charting import player_serve_profile
from modules.data_update.weather import weather_serve_adjustment
from modules.feature_engineering.elo import EloEngine, expected_score
from modules.markov.chain import estimate_serve_probs, match_win_prob

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "models" / "best_model.joblib"


class MatchPredictor:
    def __init__(self):
        self.elo = EloEngine()
        self.bundle = None
        if MODEL_PATH.exists():
            self.bundle = joblib.load(MODEL_PATH)

    def predict_match(
        self,
        player_a: str,
        player_b: str,
        *,
        elo_a: float,
        elo_b: float,
        surface: str = "Hard",
        best_of: int = 3,
        surface_wr_a: float = 0.5,
        surface_wr_b: float = 0.5,
        features: dict | None = None,
        tourney_name: str | None = None,
        weather: dict | None = None,
    ) -> dict:
        """Predice P(A vince) combinando Markov + ML."""
        adj = 0.0
        adj += altitude_serve_boost(lookup_altitude(tourney_name))
        adj += cpi_serve_adjustment(lookup_cpi(tourney_name, surface=surface))
        adj += weather_serve_adjustment(weather)
        adj += skill_serve_adjustment(lookup_player_skills(player_a))
        adj -= skill_serve_adjustment(lookup_player_skills(player_b)) * 0.5

        mcp_a = player_serve_profile(player_a)
        mcp_b = player_serve_profile(player_b)
        if mcp_a and mcp_b:
            p_hold_a = mcp_a.get("p_first_serve_won", 0.65)
            p_hold_b = mcp_b.get("p_first_serve_won", 0.65)
            adj += (p_hold_a - p_hold_b) * 0.05

        p_serve_a, p_serve_b = estimate_serve_probs(
            elo_a, elo_b,
            surface_wr_a=surface_wr_a,
            surface_wr_b=surface_wr_b,
            adjustments=adj,
        )
        p_markov = match_win_prob(p_serve_a, p_serve_b, best_of=best_of)
        p_elo = expected_score(elo_a, elo_b)

        p_ml = None
        if self.bundle and features:
            cols = self.bundle["feature_cols"]
            X = np.array([[features.get(c, 0.0) for c in cols]])
            p_ml = float(self.bundle["model"].predict_proba(X)[0, 1])

        weights = []
        probs = []
        if p_ml is not None:
            weights.append(0.35)
            probs.append(p_ml)
        weights.append(0.40)
        probs.append(p_markov)
        weights.append(0.25)
        probs.append(p_elo)

        w_sum = sum(weights)
        p_blend = sum(w * p for w, p in zip(weights, probs)) / w_sum

        return {
            "player_a": player_a,
            "player_b": player_b,
            "surface": surface,
            "best_of": best_of,
            "p_win_a": round(p_blend, 4),
            "p_markov": round(p_markov, 4),
            "p_elo": round(p_elo, 4),
            "p_ml": round(p_ml, 4) if p_ml else None,
            "p_serve_a": round(p_serve_a, 4),
            "p_serve_b": round(p_serve_b, 4),
            "components": {
                "markov": p_markov,
                "elo": p_elo,
                "ml": p_ml,
            },
        }
