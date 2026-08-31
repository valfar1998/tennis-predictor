"""Build upcoming predictions: Betfair live odds + model → value check."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from modules.advisor.advise import advise
from modules.advisor.playability import enrich_playability, MIN_PLAY_ALERT
from modules.data_update.entity_resolution import _norm_name, build_player_index, resolve_name
from modules.data_update.history import archive_prediction
from modules.data_update.sackmann import ensure_sackmann_wta, load_tour_matches, load_wta_matches
from modules.data_update.tennis_abstract import lookup_ta_elo
from modules.feature_engineering.elo import EloEngine, expected_score
from modules.feature_engineering.live_features import build_live_features
from modules.predictor.predict import MatchPredictor

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "upcoming_predictions.json"
RAW_ATP = ROOT / "data" / "raw" / "atp"
RAW_WTA = ROOT / "data" / "raw" / "wta"


@dataclass
class TourBundle:
    tour: str
    matches: pd.DataFrame
    elo_engine: EloEngine
    name_to_id: dict[str, int]
    candidates: list[str]


def _infer_surface(competition: str) -> str:
    c = str(competition or "").lower()
    clay_kw = ("roland garros", "french open", "monte carlo", "madrid", "rome", "barcelona", "clay")
    grass_kw = ("wimbledon", "queens", "halle", "eastbourne", "grass")
    if any(k in c for k in clay_kw):
        return "Clay"
    if any(k in c for k in grass_kw):
        return "Grass"
    return "Hard"


def _infer_tour(competition: str) -> str:
    c = str(competition or "").lower()
    if any(k in c for k in ("women", " wta", "wta ", "ladies", "female")):
        return "WTA"
    if any(k in c for k in ("men", " atp", "atp ", "gentlemen")):
        return "ATP"
    return "ATP"


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


def _load_tour_bundle(*, tour: str, min_year: int) -> TourBundle | None:
    tour = tour.upper()
    matches = pd.DataFrame()
    players_path = RAW_ATP / "atp_players.csv" if tour == "ATP" else RAW_WTA / "wta_players.csv"

    try:
        if tour == "WTA":
            matches = load_wta_matches(min_year=min_year)
        else:
            matches = load_tour_matches(min_year=min_year)
    except FileNotFoundError:
        if tour == "ATP":
            return None
        from modules.data_update.tennis_abstract import load_elo_index

        if not load_elo_index(tour="wta"):
            return None

    if tour == "ATP" and matches.empty:
        return None

    name_to_id: dict[str, int] = {}
    candidates: list[str] = []
    if players_path.exists():
        players = pd.read_csv(players_path, low_memory=False)
        id_to_name = build_player_index(players)
        for pid, name in id_to_name.items():
            name_to_id[_norm_name(name)] = pid
            candidates.append(name)

    if tour == "WTA":
        chart_path = ROOT / "data" / "raw" / "charting" / "charting-w-matches.csv"
        if chart_path.exists():
            wta = pd.read_csv(chart_path, usecols=["Player 1", "Player 2"], low_memory=False)
            for col in ("Player 1", "Player 2"):
                for name in wta[col].dropna().astype(str).unique():
                    name = name.strip()
                    if len(name) >= 4 and name not in candidates:
                        candidates.append(name)

    return TourBundle(
        tour=tour,
        matches=matches,
        elo_engine=_build_elo_engine(matches) if not matches.empty else EloEngine(),
        name_to_id=name_to_id,
        candidates=candidates,
    )


def _all_candidates(bundles: dict[str, TourBundle]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for bundle in bundles.values():
        for name in bundle.candidates:
            key = _norm_name(name)
            if key not in seen:
                seen.add(key)
                out.append(name)
    return out


def _pick_bundle(bundles: dict[str, TourBundle], competition: str) -> TourBundle:
    tour = _infer_tour(competition)
    if tour in bundles:
        return bundles[tour]
    return bundles.get("ATP") or next(iter(bundles.values()))


def _player_resolved(
    name: str,
    *,
    candidates: list[str],
    bundle: TourBundle,
    ta_elo: float | None,
) -> bool:
    resolved = resolve_name(name, candidates=candidates)
    pid = bundle.name_to_id.get(_norm_name(resolved))
    if pid and pid in bundle.elo_engine.players:
        return True
    return ta_elo is not None


def _player_elo(
    bundle: TourBundle,
    name: str,
    *,
    surface: str,
    candidates: list[str],
    tourney_name: str | None = None,
) -> float:
    from modules.data_update.cpi import lookup_cpi

    resolved = resolve_name(name, candidates=candidates)
    pid = bundle.name_to_id.get(_norm_name(resolved))
    ta = lookup_ta_elo(resolved, surface, tour=bundle.tour)
    cpi = lookup_cpi(tourney_name or "", surface=surface)

    if pid and pid in bundle.elo_engine.players:
        pe = bundle.elo_engine.players[pid]
        elo = pe.blended_with_cpi(surface, cpi) if cpi else pe.blended(surface)
        if ta:
            return 0.6 * elo + 0.4 * ta
        return elo
    if ta:
        return ta
    return 1500.0


def _predict_from_betfair(
    bundles: dict[str, TourBundle],
    betfair_events: list[dict],
    predictor: MatchPredictor,
    *,
    moneyway_rows: list[dict] | None = None,
    dropping_rows: list[dict] | None = None,
) -> list[dict]:
    predictions: list[dict] = []
    seen: set[str] = set()
    all_cands = _all_candidates(bundles)

    for ev in betfair_events:
        player_a = str(ev.get("player_a") or "").strip()
        player_b = str(ev.get("player_b") or "").strip()
        odd_a, odd_b = ev.get("odd_a"), ev.get("odd_b")
        if not player_a or not player_b or not odd_a or not odd_b:
            continue
        if float(odd_a) <= 1.01 or float(odd_b) <= 1.01:
            continue

        key = "|".join(sorted([_norm_name(player_a), _norm_name(player_b)]))
        if key in seen:
            continue
        seen.add(key)

        competition = str(ev.get("competition") or "")
        bundle = _pick_bundle(bundles, competition)
        tour = bundle.tour
        surface = _infer_surface(competition)

        elo_a = _player_elo(bundle, player_a, surface=surface, candidates=all_cands, tourney_name=competition)
        elo_b = _player_elo(bundle, player_b, surface=surface, candidates=all_cands, tourney_name=competition)

        ta_a = lookup_ta_elo(player_a, surface, tour=tour)
        ta_b = lookup_ta_elo(player_b, surface, tour=tour)

        resolved_a = _player_resolved(
            player_a, candidates=all_cands, bundle=bundle, ta_elo=ta_a,
        )
        resolved_b = _player_resolved(
            player_b, candidates=all_cands, bundle=bundle, ta_elo=ta_b,
        )

        pid_a = bundle.name_to_id.get(_norm_name(resolve_name(player_a, candidates=all_cands)))
        pid_b = bundle.name_to_id.get(_norm_name(resolve_name(player_b, candidates=all_cands)))
        live_feat = build_live_features(
            player_a=player_a,
            player_b=player_b,
            pid_a=pid_a,
            pid_b=pid_b,
            elo_a=elo_a,
            elo_b=elo_b,
            surface=surface,
            best_of=3,
            tourney_name=competition,
            matches=bundle.matches,
        )

        pred = predictor.predict_match(
            player_a,
            player_b,
            elo_a=elo_a,
            elo_b=elo_b,
            surface=surface,
            best_of=3,
            surface_wr_a=float(live_feat.get("surface_wr_a") or 0.5),
            surface_wr_b=float(live_feat.get("surface_wr_b") or 0.5),
            tourney_name=competition,
            features=live_feat,
            tour=tour,
        )
        pred["date"] = str(ev.get("commence_time") or "")[:10]
        pred["tourney"] = ev.get("competition")
        pred["tour"] = tour
        pred["betfair_event_id"] = ev.get("event_id")
        pred["model_low_confidence"] = not (resolved_a and resolved_b)
        pred["players_resolved"] = {"a": resolved_a, "b": resolved_b}

        advised = advise(pred, float(odd_a), float(odd_b), source="betfair")
        advised["odds_source"] = "betfair"
        advised["book_odds"] = {"a": odd_a, "b": odd_b}
        advised = enrich_playability(
            advised,
            moneyway_rows=moneyway_rows,
            dropping_rows=dropping_rows,
        )
        predictions.append(advised)
        if advised.get("action") == "bet" and (advised.get("playability") or 0) >= MIN_PLAY_ALERT:
            archive_prediction(advised)

    return predictions


def _predict_from_sackmann_recent(
    bundle: TourBundle,
    predictor: MatchPredictor,
    *,
    moneyway_rows: list[dict] | None = None,
    dropping_rows: list[dict] | None = None,
    all_candidates: list[str] | None = None,
) -> list[dict]:
    """Fallback: match recenti Sackmann con quote storiche mergeate."""
    matches = bundle.matches
    clean = matches[~matches["score"].astype(str).str.contains("W/O|RET|DEF", case=False, na=False)]
    cutoff = datetime.now() - timedelta(days=3)
    recent = clean[pd.to_datetime(clean["tourney_date"]) >= cutoff]
    predictions: list[dict] = []
    seen: set[tuple[str, str]] = set()
    cands = all_candidates or bundle.candidates

    for _, row in recent.iterrows():
        wname, lname = str(row["winner_name"]), str(row["loser_name"])
        key = tuple(sorted([wname, lname]))
        if key in seen:
            continue
        seen.add(key)

        wid, lid = int(row["winner_id"]), int(row["loser_id"])
        surface = str(row.get("surface") or "Hard")
        pe_w = bundle.elo_engine.players.get(wid)
        pe_l = bundle.elo_engine.players.get(lid)
        elo_w = pe_w.blended(surface) if pe_w else 1500.0
        elo_l = pe_l.blended(surface) if pe_l else 1500.0

        ta_w = lookup_ta_elo(wname, surface, tour=bundle.tour)
        ta_l = lookup_ta_elo(lname, surface, tour=bundle.tour)
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
        pred["tour"] = bundle.tour
        pred["tourney_level"] = row.get("tourney_level")
        pred["model_low_confidence"] = not (
            wid in bundle.elo_engine.players and lid in bundle.elo_engine.players
        )

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

        advised = enrich_playability(
            advised,
            moneyway_rows=moneyway_rows,
            dropping_rows=dropping_rows,
        )
        predictions.append(advised)
        if advised.get("action") == "bet" and (advised.get("playability") or 0) >= MIN_PLAY_ALERT:
            archive_prediction(advised)

    return predictions


def build_upcoming(*, days_ahead: int = 14, use_betfair: bool = True) -> list[dict]:
    """Genera predizioni e value bet vs quote bookmaker (Betfair Exchange)."""
    min_year = datetime.now().year - 1
    ensure_sackmann_wta(clone=True)
    from modules.data_update.tennis_abstract import fetch_tennis_abstract_elo

    fetch_tennis_abstract_elo(tour="wta", force=False)
    fetch_tennis_abstract_elo(tour="atp", force=False)

    bundles: dict[str, TourBundle] = {}
    for tour in ("ATP", "WTA"):
        bundle = _load_tour_bundle(tour=tour, min_year=min_year)
        if bundle:
            bundles[tour] = bundle

    if not bundles:
        return []

    from modules.data_update.market_signals import sync_market_signals, load_dropping_cache, load_moneyway_cache

    sync_market_signals(force=False)
    moneyway_rows = load_moneyway_cache()
    dropping_rows = load_dropping_cache()

    predictor = MatchPredictor()
    predictions: list[dict] = []

    if use_betfair:
        try:
            from modules.data_update.betfair import fetch_betfair_odds, load_betfair_cache

            bf = fetch_betfair_odds(days=min(days_ahead, 7), max_age_hours=1.0)
            events = bf.get("events") if bf.get("ok") else load_betfair_cache()
            if events:
                predictions = _predict_from_betfair(
                    bundles,
                    events,
                    predictor,
                    moneyway_rows=moneyway_rows,
                    dropping_rows=dropping_rows,
                )
        except Exception as exc:
            print(f"betfair skip: {exc}")

    if not predictions:
        all_cands = _all_candidates(bundles)
        for bundle in bundles.values():
            predictions.extend(
                _predict_from_sackmann_recent(
                    bundle,
                    predictor,
                    moneyway_rows=moneyway_rows,
                    dropping_rows=dropping_rows,
                    all_candidates=all_cands,
                )
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(predictions, indent=2, default=str), encoding="utf-8")
    return predictions
