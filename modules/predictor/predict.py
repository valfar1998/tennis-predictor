"""Match predictor: blend Elo + Markov (pressure) + ML + meta-learner."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from modules.data_update.altitude import lookup_altitude
from modules.data_update.charting import player_pressure_profile, player_serve_profile
from modules.data_update.cpi import cpi_serve_adjustment, effective_cpi
from modules.data_update.tennisratio import lookup_player_skills, skill_serve_adjustment
from modules.data_update.weather import weather_serve_adjustment
from modules.feature_engineering.air_density import air_density_kg_m3, serve_adjustment_from_air
from modules.feature_engineering.elo import EloEngine, expected_score
from modules.markov.pressure import estimate_serve_probs_full
from modules.model_training.stacker import MetaStacker

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "models" / "best_model.joblib"


class MatchPredictor:
    def __init__(self):
        self.elo = EloEngine()
        self.bundle = None
        self.stacker = MetaStacker()
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
        tour: str = "ATP",
        serve_elo_a: float | None = None,
        return_elo_a: float | None = None,
        serve_elo_b: float | None = None,
        return_elo_b: float | None = None,
    ) -> dict:
        # Elo già CPI-modulato upstream; optional refinement da feature CPI
        p_elo = expected_score(elo_a, elo_b)
        alt_m = lookup_altitude(tourney_name)
        cpi_eff = effective_cpi(tourney_name or "", surface=surface, weather=weather, altitude_m=alt_m)
        cpi_norm = float((features or {}).get("cpi_norm") or cpi_eff or 1.0)

        adj = 0.0
        if weather:
            rho = air_density_kg_m3(
                temp_c=float(weather.get("temp_c") or 22),
                humidity_pct=float(weather.get("humidity_pct") or 50),
                pressure_hpa=weather.get("pressure_hpa"),
                altitude_m=alt_m,
            )
            adj += serve_adjustment_from_air(rho)
        adj += cpi_serve_adjustment(cpi_norm)
        adj += weather_serve_adjustment(weather)
        adj += skill_serve_adjustment(lookup_player_skills(player_a))
        adj -= skill_serve_adjustment(lookup_player_skills(player_b)) * 0.5

        mcp_a = player_serve_profile(player_a, tour="w" if tour == "WTA" else "m")
        mcp_b = player_serve_profile(player_b, tour="w" if tour == "WTA" else "m")
        if mcp_a and mcp_b:
            p_hold_a = mcp_a.get("p_first_serve_won", 0.65)
            p_hold_b = mcp_b.get("p_first_serve_won", 0.65)
            adj += (p_hold_a - p_hold_b) * 0.05

        pressure_a = (features or {}).get("_pressure_a") or player_pressure_profile(
            player_a, tour="w" if tour == "WTA" else "m"
        )
        pressure_b = (features or {}).get("_pressure_b") or player_pressure_profile(
            player_b, tour="w" if tour == "WTA" else "m"
        )

        markov = estimate_serve_probs_full(
            elo_a,
            elo_b,
            surface_wr_a=surface_wr_a,
            surface_wr_b=surface_wr_b,
            adjustments=adj,
            pressure_a=pressure_a,
            pressure_b=pressure_b,
            best_of=best_of,
            serve_elo_a=serve_elo_a,
            return_elo_a=return_elo_a,
            serve_elo_b=serve_elo_b,
            return_elo_b=return_elo_b,
            cpi_factor=cpi_norm,
        )
        p_markov = markov["p_markov"]
        p_serve_a = markov["p_serve_a"]
        p_serve_b = markov["p_serve_b"]

        p_ml = None
        if self.bundle and features:
            cols = self.bundle["feature_cols"]
            X = np.array([[float(features.get(c, 0.0) or 0.0) for c in cols]])
            p_ml = float(self.bundle["model"].predict_proba(X)[0, 1])

        p_blend = self.stacker.predict(p_markov=p_markov, p_elo=p_elo, p_ml=p_ml)

        return {
            "player_a": player_a,
            "player_b": player_b,
            "surface": surface,
            "best_of": best_of,
            "tour": tour,
            "p_win_a": round(p_blend, 4),
            "p_markov": round(p_markov, 4),
            "p_elo": round(p_elo, 4),
            "p_ml": round(p_ml, 4) if p_ml else None,
            "p_serve_a": p_serve_a,
            "p_serve_b": p_serve_b,
            "cpi_norm": cpi_norm,
            "cpi_effective": cpi_norm,
            "serve_return_elo": {
                "serve_a": serve_elo_a,
                "return_a": return_elo_a,
                "serve_b": serve_elo_b,
                "return_b": return_elo_b,
            },
            "pressure_used": bool(pressure_a or pressure_b),
            "components": {
                "markov": p_markov,
                "elo": p_elo,
                "ml": p_ml,
                "stacker_weights": self.stacker.weights,
            },
        }
