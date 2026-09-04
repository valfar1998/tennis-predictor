"""Build upcoming predictions: Betfair live odds + model → value check."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from modules.advisor.advise import advise
from modules.advisor.playability import enrich_playability
from modules.data_update.entity_resolution import _norm_name, build_player_index, resolve_name
from modules.data_update.history import archive_prediction, paper_eligible
from modules.data_update.sackmann import ensure_sackmann_wta, load_tour_matches, load_wta_matches
from modules.data_update.tennis_abstract import lookup_ta_elo
from modules.data_update.tennis_livescore import fetch_tennis_livescore
from modules.feature_engineering.elo import EloEngine, expected_score
from modules.feature_engineering.live_features import build_live_features
from modules.feature_engineering.serve_return_elo import ServeReturnEloEngine
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
    sr_elo_engine: ServeReturnEloEngine
    name_to_id: dict[str, int]
    candidates: list[str]
    player_dob: dict[int, int]


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


def _build_sr_elo_engine(matches: pd.DataFrame) -> ServeReturnEloEngine:
    engine = ServeReturnEloEngine()
    clean = matches[~matches["score"].astype(str).str.contains("W/O|RET|DEF", case=False, na=False)]
    for _, row in clean.sort_values("tourney_date").iterrows():
        try:
            engine.update_from_match(
                int(row["winner_id"]),
                int(row["loser_id"]),
                row,
                surface=str(row.get("surface") or "Hard"),
                level=row.get("tourney_level"),
                match_date=pd.Timestamp(row["tourney_date"]).to_pydatetime(),
            )
        except Exception:
            continue
    return engine


def _load_player_dob(players_path: Path) -> dict[int, int]:
    if not players_path.is_file():
        return {}
    df = pd.read_csv(players_path, usecols=["player_id", "dob"], low_memory=False)
    out: dict[int, int] = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("player_id")) and pd.notna(row.get("dob")):
            try:
                out[int(row["player_id"])] = int(row["dob"])
            except (TypeError, ValueError):
                continue
    return out


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
        sr_elo_engine=_build_sr_elo_engine(matches) if not matches.empty else ServeReturnEloEngine(),
        name_to_id=name_to_id,
        candidates=candidates,
        player_dob=_load_player_dob(players_path),
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


def _sr_elo_for_player(
    bundle: TourBundle,
    pid: int | None,
    surface: str,
) -> tuple[float | None, float | None]:
    if pid is None or pid not in bundle.sr_elo_engine.players:
        return None, None
    pe = bundle.sr_elo_engine.players[pid]
    w = bundle.sr_elo_engine.surface_weight
    return pe.serve_blended(surface, w), pe.return_blended(surface, w)


def _retirement_context_for_pick(
    bundle: TourBundle,
    *,
    pid: int | None,
    live_feat: dict,
    pick_prob: float,
) -> dict:
    from modules.advisor.retirement_risk import _player_age_years, estimate_retirement_risk, historical_retirement_rate

    age = None
    if pid and pid in bundle.player_dob:
        age = _player_age_years(bundle.player_dob[pid])
    fatigue = float(live_feat.get("fatigue_minutes_7d_a") or live_feat.get("fatigue_minutes_7d_b") or 0)
    rest = float(live_feat.get("rest_days_a") or live_feat.get("rest_days_b") or 7)
    hist = historical_retirement_rate(bundle.matches, pid) if pid else 0.0
    p_retire = estimate_retirement_risk(
        age=age,
        fatigue_minutes_72h=fatigue * 3,
        rest_days=rest,
        historical_retire_rate=hist,
        is_favorite=pick_prob >= 0.55,
    )
    return {"player_injury_risk": p_retire, "p_retire": p_retire}


def _player_elo(
    bundle: TourBundle,
    name: str,
    *,
    surface: str,
    candidates: list[str],
    tourney_name: str | None = None,
    match_date: str | None = None,
) -> tuple[float, dict | None]:
    from modules.data_update.cpi import lookup_cpi
    from modules.constants import ELO_SURFACE_WEIGHT
    from modules.data_update.entity_resolution import dataset_match_count, shrink_rating_low_sample
    from modules.feature_engineering.surface_transition import (
        transition_context,
        transition_surface_weight,
    )

    resolved = resolve_name(name, candidates=candidates)
    pid = bundle.name_to_id.get(_norm_name(resolved))
    ta = lookup_ta_elo(resolved, surface, tour=bundle.tour)
    cpi = lookup_cpi(tourney_name or "", surface=surface)
    surf_w = transition_surface_weight(ELO_SURFACE_WEIGHT, surface, match_date)
    trans = transition_context(surface, match_date)

    if pid and pid in bundle.elo_engine.players:
        pe = bundle.elo_engine.players[pid]
        elo = pe.blended_with_cpi(surface, cpi, base_weight=surf_w) if cpi else pe.blended(surface, surf_w)
        n_ds = dataset_match_count(pid, bundle.matches)
        elo = shrink_rating_low_sample(elo, n_ds)
        trans = {**trans, "n_dataset": n_ds, "elo_shrunk": n_ds < 15}
        if ta:
            return 0.6 * elo + 0.4 * ta, trans
        return elo, trans
    if ta:
        return ta, trans
    return 1500.0, trans


def _predict_from_betfair(
    bundles: dict[str, TourBundle],
    betfair_events: list[dict],
    predictor: MatchPredictor,
    *,
    moneyway_rows: list[dict] | None = None,
    dropping_rows: list[dict] | None = None,
    min_edge: float | None = None,
) -> list[dict]:
    predictions: list[dict] = []
    seen: set[str] = set()
    all_cands = _all_candidates(bundles)
    n_events = len(betfair_events)
    every = max(1, n_events // 10) if n_events else 1

    for idx, ev in enumerate(betfair_events, 1):
        if idx == 1 or idx == n_events or idx % every == 0:
            from modules.ops_progress import log_item

            pa0 = str(ev.get("player_a") or "?")
            pb0 = str(ev.get("player_b") or "?")
            log_item(idx, n_events, f"analisi {pa0} vs {pb0}")
        try:
            row = _predict_one_betfair_event(
                ev,
                bundles=bundles,
                predictor=predictor,
                all_cands=all_cands,
                moneyway_rows=moneyway_rows,
                dropping_rows=dropping_rows,
                min_edge=min_edge,
                seen=seen,
            )
        except Exception as exc:
            pa = str(ev.get("player_a") or "?")
            pb = str(ev.get("player_b") or "?")
            print(f"  skip {pa} vs {pb}: {exc}", flush=True)
            continue
        if row:
            predictions.append(row)
    return predictions


def _predict_one_betfair_event(
    ev: dict,
    *,
    bundles: dict[str, TourBundle],
    predictor: MatchPredictor,
    all_cands: list[str],
    moneyway_rows: list[dict] | None,
    dropping_rows: list[dict] | None,
    min_edge: float | None,
    seen: set[str],
) -> dict | None:
    player_a = str(ev.get("player_a") or "").strip()
    player_b = str(ev.get("player_b") or "").strip()
    odd_a, odd_b = ev.get("odd_a"), ev.get("odd_b")
    if not player_a or not player_b or not odd_a or not odd_b:
        return None
    if float(odd_a) <= 1.01 or float(odd_b) <= 1.01:
        return None

    from modules.data_update.entity_resolution import align_odds_to_players

    odds_aligned = align_odds_to_players(
        player_a,
        player_b,
        odd_a,
        odd_b,
        runner_a=ev.get("runner_a"),
        runner_b=ev.get("runner_b"),
    )
    if odds_aligned.get("blocked"):
        return None
    odd_a, odd_b = odds_aligned["odd_a"], odds_aligned["odd_b"]

    key = "|".join(sorted([_norm_name(player_a), _norm_name(player_b)]))
    if key in seen:
        return None
    seen.add(key)

    competition = str(ev.get("competition") or "")
    bundle = _pick_bundle(bundles, competition)
    tour = bundle.tour
    surface = _infer_surface(competition)

    elo_a, trans_a = _player_elo(
        bundle, player_a, surface=surface, candidates=all_cands,
        tourney_name=competition, match_date=str(ev.get("commence_time") or "")[:10],
    )
    elo_b, trans_b = _player_elo(
        bundle, player_b, surface=surface, candidates=all_cands,
        tourney_name=competition, match_date=str(ev.get("commence_time") or "")[:10],
    )

    ta_a = lookup_ta_elo(player_a, surface, tour=tour)
    ta_b = lookup_ta_elo(player_b, surface, tour=tour)

    resolved_a = _player_resolved(
        player_a, candidates=all_cands, bundle=bundle, ta_elo=ta_a,
    )
    resolved_b = _player_resolved(
        player_b, candidates=all_cands, bundle=bundle, ta_elo=ta_b,
    )

    match_date = str(ev.get("commence_time") or "")[:10]
    resolved_a_name = resolve_name(
        player_a, candidates=all_cands, opponent_name=player_b,
        tourney_date=match_date, tour=tour,
    )
    resolved_b_name = resolve_name(
        player_b, candidates=all_cands, opponent_name=player_a,
        tourney_date=match_date, tour=tour,
    )
    pid_a = bundle.name_to_id.get(_norm_name(resolved_a_name))
    pid_b = bundle.name_to_id.get(_norm_name(resolved_b_name))
    serve_a, ret_a = _sr_elo_for_player(bundle, pid_a, surface)
    serve_b, ret_b = _sr_elo_for_player(bundle, pid_b, surface)

    weather = None
    try:
        from modules.data_update.weather import geocode_city, fetch_weather

        city_guess = competition.split()[-1] if competition else ""
        geo = geocode_city(city_guess)
        if geo and ev.get("commence_time"):
            when = pd.Timestamp(ev.get("commence_time")).to_pydatetime()
            weather = fetch_weather(geo["lat"], geo["lon"], when)
    except Exception:
        weather = None

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
        weather=weather,
        serve_elo_a=serve_a,
        return_elo_a=ret_a,
        serve_elo_b=serve_b,
        return_elo_b=ret_b,
    )

    from modules.data_update.entity_resolution import detect_model_odds_inversion

    if detect_model_odds_inversion(pred.get("p_win_a", 0.5), odd_a, odd_b):
        odd_a, odd_b = odd_b, odd_a
        pred["odds_inversion_corrected"] = True
        odds_aligned = align_odds_to_players(
            player_a, player_b, odd_a, odd_b,
            runner_a=player_a, runner_b=player_b,
        )
        odd_a, odd_b = odds_aligned["odd_a"], odds_aligned["odd_b"]

    pred["date"] = match_date
    pred["tourney"] = ev.get("competition")
    if ev.get("commence_time"):
        pred["commence_time_utc"] = str(ev.get("commence_time"))
        try:
            pred["_match_start_dt"] = pd.Timestamp(ev.get("commence_time")).to_pydatetime()
        except Exception:
            pass
    from modules.data_update.calendar_utils import apply_local_schedule

    apply_local_schedule(pred, commence_time=ev.get("commence_time"))
    from modules.advisor.risk_controls import infer_tourney_level

    pred["tourney_level"] = infer_tourney_level(pred["tourney"])
    pred["tour"] = tour
    pred["betfair_event_id"] = ev.get("event_id")
    pred["betfair_market_id"] = ev.get("market_id")
    pred["model_low_confidence"] = not (resolved_a and resolved_b)
    pred["players_resolved"] = {"a": resolved_a, "b": resolved_b}
    n_ds_a = int(live_feat.get("n_dataset_a") or 0)
    n_ds_b = int(live_feat.get("n_dataset_b") or 0)
    pred["data_density"] = {
        "a": n_ds_a,
        "b": n_ds_b,
        "min": min(n_ds_a, n_ds_b),
    }
    if trans_a.get("in_transition") or trans_b.get("in_transition"):
        pred["surface_transition"] = {
            "a": trans_a,
            "b": trans_b,
            "note": "peso Elo surface ridotto (primi 7 gg cambio stagione)",
        }

    if odds_aligned.get("swapped") or ev.get("odds_swapped"):
        pred["odds_swapped"] = True
    pred["odds_verified"] = odds_aligned.get("verified", ev.get("odds_verified"))
    pred["book_odds"] = {"a": odd_a, "b": odd_b}

    from modules.advisor.clv_live import resolve_close_odds

    close = resolve_close_odds(
        player_a,
        player_b,
        date=pred["date"],
        tour=tour,
        betfair_event_id=ev.get("event_id"),
        betfair_market_id=ev.get("market_id"),
        betfair_odds={"a": odd_a, "b": odd_b},
    )
    if close:
        pred["close_odds"] = close
        pred["pinnacle_odds"] = close  # backward compat

    from modules.data_update.market_signals import lookup_dropping

    drop_row = lookup_dropping(player_a, player_b, rows=dropping_rows)

    ret_ctx = {}
    if pred.get("p_win_a") is not None:
        pick_side = "A" if float(pred.get("p_win_a", 0.5)) >= 0.5 else "B"
        pick_pid = pid_a if pick_side == "A" else pid_b
        pick_prob = float(pred.get("p_win_a") if pick_side == "A" else 1 - pred.get("p_win_a", 0.5))
        feat_pick = dict(live_feat)
        if pick_side == "B":
            feat_pick = {
                **live_feat,
                "fatigue_minutes_7d_a": live_feat.get("fatigue_minutes_7d_b"),
                "rest_days_a": live_feat.get("rest_days_b"),
            }
        ret_ctx = _retirement_context_for_pick(
            bundle, pid=pick_pid, live_feat=feat_pick, pick_prob=pick_prob
        )

    advised = advise(
        pred,
        float(odd_a),
        float(odd_b),
        source=str(ev.get("odds_source") or "betfair"),
        dropping_row=drop_row,
        min_edge=min_edge,
        bookmaker="betfair" if str(ev.get("odds_source") or "betfair") == "betfair" else "default",
        retirement_context=ret_ctx,
    )
    advised["odds_source"] = str(ev.get("odds_source") or "betfair")
    advised["book_odds"] = {"a": odd_a, "b": odd_b}
    advised = enrich_playability(
        advised,
        moneyway_rows=moneyway_rows,
        dropping_rows=dropping_rows,
    )
    if advised.get("action") == "bet":
        archive_prediction(advised)
    else:
        if paper_eligible(advised):
            paper = dict(advised)
            paper["action"] = "paper"
            paper["recommended"] = advised.get("best_play") or (advised.get("value") or {}).get("best")
            archive_prediction(paper)
    return advised


def _predict_from_sackmann_recent(
    bundle: TourBundle,
    predictor: MatchPredictor,
    *,
    moneyway_rows: list[dict] | None = None,
    dropping_rows: list[dict] | None = None,
    all_candidates: list[str] | None = None,
    min_edge: float | None = None,
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

        from modules.data_update.market_signals import lookup_dropping

        drop_row = lookup_dropping(wname, lname, rows=dropping_rows)

        if bf_match:
            advised = advise(
                pred,
                bf_match.get("odd_a"),
                bf_match.get("odd_b"),
                source="betfair",
                dropping_row=drop_row,
                min_edge=min_edge,
            )
            advised["odds_source"] = "betfair"
        else:
            advised = advise(pred, ow, ol, source="book", dropping_row=drop_row, min_edge=min_edge)
            advised["odds_source"] = "book"

        advised = enrich_playability(
            advised,
            moneyway_rows=moneyway_rows,
            dropping_rows=dropping_rows,
        )
        predictions.append(advised)
        if advised.get("action") == "bet":
            archive_prediction(advised)
        else:
            if paper_eligible(advised):
                paper = dict(advised)
                paper["action"] = "paper"
                paper["recommended"] = advised.get("best_play") or (advised.get("value") or {}).get("best")
                archive_prediction(paper)

    return predictions


def _merge_predictions(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Unisce predizioni senza duplicare lo stesso match (Betfair ha priorità)."""
    from modules.data_update.entity_resolution import _norm_name

    out = list(primary)
    seen = {
        "|".join(sorted([_norm_name(p.get("player_a", "")), _norm_name(p.get("player_b", ""))]))
        for p in primary
        if p.get("player_a") and p.get("player_b")
    }
    added = 0
    for pred in secondary:
        key = "|".join(sorted([_norm_name(pred.get("player_a", "")), _norm_name(pred.get("player_b", ""))]))
        if key in seen:
            continue
        out.append(pred)
        seen.add(key)
        added += 1
    if added:
        print(f"  upcoming: +{added} predizioni da Kambi/Unibet", flush=True)
    return out


def build_upcoming(*, days_ahead: int = 14, use_betfair: bool = True) -> list[dict]:
    """Genera predizioni e value bet vs quote bookmaker (Betfair Exchange)."""
    from modules.ops_progress import OpProgress, log_done

    min_year = datetime.now().year - 1
    from modules.data_update.sackmann import ensure_sackmann_atp, ensure_sackmann_wta

    prog = OpProgress(10, label="upcoming")
    prog.next("Sync Sackmann ATP/WTA...")
    ensure_sackmann_atp(clone=True)
    ensure_sackmann_wta(clone=True)
    from modules.data_update.tennis_abstract import fetch_tennis_abstract_elo

    prog.next("Tennis Abstract Elo...")
    fetch_tennis_abstract_elo(tour="wta", force=False)
    fetch_tennis_abstract_elo(tour="atp", force=False)

    bundles: dict[str, TourBundle] = {}
    prog.next("Carica bundle tour ATP/WTA...")
    for tour in ("ATP", "WTA"):
        bundle = _load_tour_bundle(tour=tour, min_year=min_year)
        if bundle:
            bundles[tour] = bundle

    if not bundles:
        print("  upcoming: nessun bundle tour disponibile", flush=True)
        return []

    prog.next("Player graph + biografiche...")
    try:
        from modules.data_update.player_graph import build_match_edges, sync_player_biographics

        for tour, bundle in bundles.items():
            players_path = RAW_ATP / "atp_players.csv" if tour == "ATP" else RAW_WTA / "wta_players.csv"
            if players_path.is_file():
                sync_player_biographics(pd.read_csv(players_path, low_memory=False), tour=tour)
            if not bundle.matches.empty:
                build_match_edges(bundle.matches, tour=tour, limit=8000)
    except Exception as exc:
        print(f"  player graph skip: {exc}", flush=True)

    from modules.data_update.market_signals import sync_market_signals, load_dropping_cache, load_moneyway_cache
    from modules.data_update.tennis_data_odds import download_tennis_data_odds

    prog.next("Dati mercato (tennis-data, livescore, signals)...")
    try:
        download_tennis_data_odds(force=False)
    except Exception as exc:
        print(f"  tennis-data skip: {exc}", flush=True)
    try:
        fetch_tennis_livescore(force=False)
    except Exception as exc:
        print(f"  livescore skip: {exc}", flush=True)

    sync_market_signals(force=False)
    moneyway_rows = load_moneyway_cache()
    dropping_rows = load_dropping_cache()

    from modules.advisor.risk_controls import apply_daily_exposure_limits, get_risk_context

    risk_ctx = get_risk_context()
    min_edge = float(risk_ctx["min_edge"])

    predictor = MatchPredictor()
    predictions: list[dict] = []

    if use_betfair:
        try:
            from modules.data_update.betfair import fetch_betfair_odds, load_betfair_cache

            prog.next("Predizioni da Betfair...")
            bf = fetch_betfair_odds(days=min(days_ahead, 7), max_age_hours=1.0)
            events = bf.get("events") if bf.get("ok") else load_betfair_cache()
            if events:
                predictions = _predict_from_betfair(
                    bundles,
                    events,
                    predictor,
                    moneyway_rows=moneyway_rows,
                    dropping_rows=dropping_rows,
                    min_edge=min_edge,
                )
        except Exception as exc:
            print(f"  betfair skip: {exc}", flush=True)

    try:
        from modules.data_update.kambi_unibet import fetch_kambi_tennis_odds, load_kambi_cache, merge_odds_events

        prog.next("Palinsesto Kambi/Unibet (ITF/Challenger)...")
        kambi = fetch_kambi_tennis_odds(force=False)
        kambi_events = kambi.get("events") if kambi.get("ok") else load_kambi_cache()
        if kambi_events:
            merged = merge_odds_events([], kambi_events)
            print(
                f"  Kambi: {len(kambi_events)} eventi "
                f"(ITF {kambi.get('n_itf', sum(1 for e in kambi_events if e.get('is_itf')))}, "
                f"Challenger {kambi.get('n_challenger', sum(1 for e in kambi_events if e.get('is_challenger')))})",
                flush=True,
            )
            if merged:
                kambi_preds = _predict_from_betfair(
                    bundles,
                    merged,
                    predictor,
                    moneyway_rows=moneyway_rows,
                    dropping_rows=dropping_rows,
                    min_edge=min_edge,
                )
                predictions = _merge_predictions(predictions, kambi_preds)
    except Exception as exc:
        print(f"  kambi/unibet skip: {exc}", flush=True)

    if not predictions:
        prog.next("Fallback predizioni Sackmann recenti...")
        all_cands = _all_candidates(bundles)
        for bundle in bundles.values():
            predictions.extend(
                _predict_from_sackmann_recent(
                    bundle,
                    predictor,
                    moneyway_rows=moneyway_rows,
                    dropping_rows=dropping_rows,
                    all_candidates=all_cands,
                    min_edge=min_edge,
                )
            )

    prog.next("Limiti esposizione + salvataggio...")
    predictions = apply_daily_exposure_limits(predictions)
    for pred in predictions:
        pred["risk_session"] = {
            "min_edge": min_edge,
            "circuit_breaker": risk_ctx["circuit_breaker"]["active"],
        }

    from modules.data_update.calendar_utils import normalize_predictions_calendar

    predictions = normalize_predictions_calendar(predictions)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(predictions, indent=2, default=str), encoding="utf-8")
    log_done(f"upcoming: {len(predictions)} predizioni, {sum(1 for p in predictions if p.get('action')=='bet')} bet")
    return predictions
