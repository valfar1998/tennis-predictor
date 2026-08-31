"""Build upcoming predictions: Betfair live odds + model → value check."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from modules.advisor.advise import advise
from modules.data_update.entity_resolution import _norm_name, build_player_index, resolve_name
from modules.data_update.history import archive_prediction
from modules.data_update.sackmann import load_tour_matches
from modules.data_update.tennis_abstract import lookup_ta_elo
from modules.feature_engineering.elo import EloEngine
from modules.predictor.predict import MatchPredictor

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "upcoming_predictions.json"
RAW_ATP = ROOT / "data" / "raw" / "atp"


def _infer_surface(competition: str) -> str:
    c = str(competition or "").lower()
    clay_kw = ("roland garros", "french open", "monte carlo", "madrid", "rome", "barcelona", "clay")
    grass_kw = ("wimbledon", "queens", "halle", "eastbourne", "grass")
    if any(k in c for k in clay_kw):
        return "Clay"
    if any(k in c for k in grass_kw):
        return "Grass"
    return "Hard"


def _build_elo_engine(matches: pd.DataFrame) -> EloEngine:
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
    return elo_engine


def _load_player_lookup() -> tuple[dict[int, str], dict[str, int], list[str]]:
    """Ritorna id→nome, nome_norm→id, lista nomi candidati."""
    path = RAW_ATP / "atp_players.csv"
    if not path.exists():
        return {}, {}, []
    players = pd.read_csv(path, low_memory=False)
    id_to_name = build_player_index(players)
    name_to_id: dict[str, int] = {}
    for pid, name in id_to_name.items():
        name_to_id[_norm_name(name)] = pid
    candidates = list(id_to_name.values())
    return id_to_name, name_to_id, candidates


def _player_elo(
    elo_engine: EloEngine,
    name: str,
    *,
    surface: str,
    candidates: list[str],
    name_to_id: dict[str, int],
) -> float:
    resolved = resolve_name(name, candidates=candidates)
    pid = name_to_id.get(_norm_name(resolved))
    if pid and pid in elo_engine.players:
        return elo_engine.players[pid].blended(surface)
    return 1500.0


def _predict_from_betfair(
    elo_engine: EloEngine,
    betfair_events: list[dict],
    predictor: MatchPredictor,
    *,
    candidates: list[str],
    name_to_id: dict[str, int],
) -> list[dict]:
    predictions: list[dict] = []
    seen: set[str] = set()

    for ev in betfair_events:
        player_a = str(ev.get("player_a") or "").strip()
        player_b = str(ev.get("player_b") or "").strip()
        odd_a, odd_b = ev.get("odd_a"), ev.get("odd_b")
        if not player_a or not player_b or not odd_a or not odd_b:
            continue

        key = "|".join(sorted([_norm_name(player_a), _norm_name(player_b)]))
        if key in seen:
            continue
        seen.add(key)

        surface = _infer_surface(ev.get("competition", ""))
        elo_a = _player_elo(elo_engine, player_a, surface=surface, candidates=candidates, name_to_id=name_to_id)
        elo_b = _player_elo(elo_engine, player_b, surface=surface, candidates=candidates, name_to_id=name_to_id)

        ta_a = lookup_ta_elo(player_a, surface)
        ta_b = lookup_ta_elo(player_b, surface)
        if ta_a:
            elo_a = 0.6 * elo_a + 0.4 * ta_a
        if ta_b:
            elo_b = 0.6 * elo_b + 0.4 * ta_b

        pred = predictor.predict_match(
            player_a,
            player_b,
            elo_a=elo_a,
            elo_b=elo_b,
            surface=surface,
            best_of=3,
            tourney_name=str(ev.get("competition") or ""),
        )
        pred["date"] = str(ev.get("commence_time") or "")[:10]
        pred["tourney"] = ev.get("competition")
        pred["betfair_event_id"] = ev.get("event_id")

        advised = advise(pred, float(odd_a), float(odd_b), source="betfair")
        advised["odds_source"] = "betfair"
        advised["book_odds"] = {"a": odd_a, "b": odd_b}
        predictions.append(advised)
        if advised.get("action") == "bet":
            archive_prediction(advised)

    return predictions


def _predict_from_sackmann_recent(
    elo_engine: EloEngine,
    matches: pd.DataFrame,
    predictor: MatchPredictor,
) -> list[dict]:
    """Fallback: match recenti Sackmann con quote storiche mergeate."""
    clean = matches[~matches["score"].astype(str).str.contains("W/O|RET|DEF", case=False, na=False)]
    cutoff = datetime.now() - timedelta(days=3)
    recent = clean[pd.to_datetime(clean["tourney_date"]) >= cutoff]
    predictions: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for _, row in recent.iterrows():
        wname, lname = str(row["winner_name"]), str(row["loser_name"])
        key = tuple(sorted([wname, lname]))
        if key in seen:
            continue
        seen.add(key)

        wid, lid = int(row["winner_id"]), int(row["loser_id"])
        surface = str(row.get("surface") or "Hard")
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
            wname,
            lname,
            elo_a=elo_w,
            elo_b=elo_l,
            surface=surface,
            best_of=int(row.get("best_of") or 3),
            tourney_name=str(row.get("tourney_name") or ""),
        )
        pred["date"] = str(row.get("tourney_date"))
        pred["tourney"] = row.get("tourney_name")
        pred["tourney_level"] = row.get("tourney_level")

        ow, ol = row.get("odds_winner"), row.get("odds_loser")
        bf_match = None
        try:
            from modules.data_update.betfair import load_betfair_cache, lookup_betfair_match

            bf_match = lookup_betfair_match(
                wname,
                lname,
                events=load_betfair_cache(),
                match_date=str(row.get("tourney_date"))[:10],
            )
        except Exception:
            pass

        if bf_match:
            advised = advise(
                pred,
                bf_match.get("odd_a"),
                bf_match.get("odd_b"),
                source="betfair",
            )
            advised["odds_source"] = "betfair"
        else:
            advised = advise(pred, ow, ol, source="book")
            advised["odds_source"] = "book"

        predictions.append(advised)
        if advised.get("action") == "bet":
            archive_prediction(advised)

    return predictions


def build_upcoming(*, days_ahead: int = 14, use_betfair: bool = True) -> list[dict]:
    """Genera predizioni e value bet vs quote bookmaker (Betfair Exchange)."""
    matches = load_tour_matches(min_year=datetime.now().year - 1)
    if matches.empty:
        return []

    elo_engine = _build_elo_engine(matches)
    predictor = MatchPredictor()
    _, name_to_id, candidates = _load_player_lookup()
    predictions: list[dict] = []

    if use_betfair:
        try:
            from modules.data_update.betfair import fetch_betfair_odds, load_betfair_cache

            bf = fetch_betfair_odds(days=min(days_ahead, 7), max_age_hours=1.0)
            events = bf.get("events") if bf.get("ok") else load_betfair_cache()
            if events:
                predictions = _predict_from_betfair(
                    elo_engine,
                    events,
                    predictor,
                    candidates=candidates,
                    name_to_id=name_to_id,
                )
        except Exception as exc:
            print(f"betfair skip: {exc}")

    if not predictions:
        predictions = _predict_from_sackmann_recent(elo_engine, matches, predictor)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(predictions, indent=2, default=str), encoding="utf-8")
    return predictions
