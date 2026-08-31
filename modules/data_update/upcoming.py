"""Build upcoming predictions da match recenti + Elo corrente."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from modules.advisor.advise import advise
from modules.data_update.history import archive_prediction
from modules.data_update.sackmann import load_tour_matches
from modules.data_update.tennis_abstract import lookup_ta_elo
from modules.feature_engineering.elo import EloEngine
from modules.predictor.predict import MatchPredictor

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "upcoming_predictions.json"


def build_upcoming(*, days_ahead: int = 14) -> list[dict]:
    """Genera predizioni per match imminenti (ultimi tornei in corso)."""
    matches = load_tour_matches(min_year=datetime.now().year - 1)
    if matches.empty:
        return []

    elo_engine = EloEngine()
    clean = matches[~matches["score"].astype(str).str.contains("W/O|RET|DEF", case=False, na=False)]
    for _, row in clean.sort_values("tourney_date").iterrows():
        try:
            elo_engine.update(
                int(row["winner_id"]),
                int(row["loser_id"]),
                surface=str(row.get("surface") or "Hard"),
                level=row.get("tourney_level"),
                match_date=pd.Timestamp(row["tourney_date"]).to_pydatetime(),
            )
        except Exception:
            continue

    cutoff = datetime.now() - timedelta(days=3)
    recent = clean[pd.to_datetime(clean["tourney_date"]) >= cutoff]
    predictor = MatchPredictor()
    predictions: list[dict] = []

    seen = set()
    for _, row in recent.iterrows():
        wid, lid = int(row["winner_id"]), int(row["loser_id"])
        wname, lname = str(row["winner_name"]), str(row["loser_name"])
        surface = str(row.get("surface") or "Hard")
        key = tuple(sorted([wname, lname]))
        if key in seen:
            continue
        seen.add(key)

        pe_w = elo_engine.players.get(wid)
        pe_l = elo_engine.players.get(lid)
        elo_w = pe_w.blended(surface) if pe_w else 1500.0
        elo_l = pe_l.blended(surface) if pe_l else 1500.0

        ta_w = lookup_ta_elo(wname, surface)
        ta_l = lookup_ta_elo(lname, surface)
        if ta_w:
            elo_w = 0.6 * elo_w + 0.4 * ta_w
        if ta_l:
            elo_l = 0.6 * elo_l + 0.4 * ta_l

        pred = predictor.predict_match(
            wname, lname,
            elo_a=elo_w, elo_b=elo_l,
            surface=surface,
            best_of=int(row.get("best_of") or 3),
            tourney_name=str(row.get("tourney_name") or ""),
        )
        pred["date"] = str(row.get("tourney_date"))
        pred["tourney"] = row.get("tourney_name")
        pred["tourney_level"] = row.get("tourney_level")

        ow, ol = row.get("odds_winner"), row.get("odds_loser")
        advised = advise(pred, ow, ol, source="book")
        predictions.append(advised)
        if advised.get("action") == "bet":
            archive_prediction(advised)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(predictions, indent=2, default=str), encoding="utf-8")
    return predictions
