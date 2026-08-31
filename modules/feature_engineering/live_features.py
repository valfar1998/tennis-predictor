"""Feature runtime per predizioni live (abilita ML + matchup avanzati)."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from modules.data_update.charting import player_pressure_profile
from modules.data_update.cpi import lookup_cpi
from modules.data_update.entity_resolution import (
    dataset_match_count,
    enrich_pressure_profile,
    shrink_rating_low_sample,
)
from modules.feature_engineering.travel import travel_km, timezone_shift_hours
from modules.feature_engineering.features import FEATURE_COLS


def _last_n_matches(matches: pd.DataFrame, pid: int, *, before, n: int = 15) -> pd.DataFrame:
    m = matches[
        (matches["tourney_date"] < before)
        & ((matches["winner_id"] == pid) | (matches["loser_id"] == pid))
    ].sort_values("tourney_date")
    return m.tail(n)


def _fatigue_minutes(recent: pd.DataFrame, pid: int, *, days: int) -> float:
    cutoff = recent["tourney_date"].max() if not recent.empty else None
    if cutoff is None:
        return 0.0
    window = cutoff - timedelta(days=days)
    sub = recent[recent["tourney_date"] >= window]
    mins = 0.0
    for _, r in sub.iterrows():
        if int(r["winner_id"]) == pid or int(r["loser_id"]) == pid:
            mins += float(r.get("minutes") or 90)
    return mins


def _surface_wr(recent: pd.DataFrame, pid: int, surface: str) -> float:
    wins, total = 0, 0
    for _, r in recent.iterrows():
        if str(r.get("surface") or "Hard") != surface:
            continue
        total += 1
        if int(r["winner_id"]) == pid:
            wins += 1
    return wins / total if total else 0.5


def _bp_rates(recent: pd.DataFrame, pid: int) -> tuple[float, float]:
    saves, breaks = [], []
    for _, r in recent.iterrows():
        if int(r["winner_id"]) == pid:
            faced, saved = r.get("w_bpFaced"), r.get("w_bpSaved")
            if pd.notna(faced) and float(faced) > 0:
                saves.append(float(saved or 0) / float(faced))
            lf, ls = r.get("l_bpFaced"), r.get("l_bpSaved")
            if pd.notna(lf) and float(lf) > 0:
                breaks.append((float(lf) - float(ls or 0)) / float(lf))
        elif int(r["loser_id"]) == pid:
            faced, saved = r.get("l_bpFaced"), r.get("l_bpSaved")
            if pd.notna(faced) and float(faced) > 0:
                saves.append(float(saved or 0) / float(faced))
            wf, ws = r.get("w_bpFaced"), r.get("w_bpSaved")
            if pd.notna(wf) and float(wf) > 0:
                breaks.append((float(wf) - float(ws or 0)) / float(wf))
    hold = sum(saves) / len(saves) if saves else 0.65
    brk = sum(breaks) / len(breaks) if breaks else 0.35
    return hold, brk


def build_live_features(
    *,
    player_a: str,
    player_b: str,
    pid_a: int | None,
    pid_b: int | None,
    elo_a: float,
    elo_b: float,
    surface: str = "Hard",
    best_of: int = 3,
    tourney_name: str | None = None,
    matches: pd.DataFrame | None = None,
    rank_a: float | None = None,
    rank_b: float | None = None,
) -> dict:
    """Costruisce dict feature allineato a FEATURE_COLS per predict live."""
    now = pd.Timestamp.now()
    m = matches if matches is not None else pd.DataFrame()
    m = m.copy()
    if not m.empty:
        m["tourney_date"] = pd.to_datetime(m["tourney_date"])

    rec_a = _last_n_matches(m, pid_a, before=now, n=20) if pid_a and not m.empty else pd.DataFrame()
    rec_b = _last_n_matches(m, pid_b, before=now, n=20) if pid_b and not m.empty else pd.DataFrame()

    hold_a, break_a = _bp_rates(rec_a, pid_a) if pid_a and not rec_a.empty else (0.65, 0.35)
    hold_b, break_b = _bp_rates(rec_b, pid_b) if pid_b and not rec_b.empty else (0.65, 0.35)

    last_t_a = str(rec_a.iloc[-1]["tourney_name"]) if not rec_a.empty else None
    last_t_b = str(rec_b.iloc[-1]["tourney_name"]) if not rec_b.empty else None

    cpi = lookup_cpi(tourney_name or "", surface=surface)
    feat = {
        "elo_diff": elo_a - elo_b,
        "e_w_elo": 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400)),
        "rank_diff": (float(rank_b or 500) - float(rank_a or 500)),
        "rank_points_diff": 0.0,
        "h2h_wins_a": 0,
        "h2h_surface_wins_a": 0,
        "fatigue_minutes_7d_a": _fatigue_minutes(rec_a, pid_a, days=7) if pid_a and not rec_a.empty else 0.0,
        "fatigue_minutes_7d_b": _fatigue_minutes(rec_b, pid_b, days=7) if pid_b and not rec_b.empty else 0.0,
        "fatigue_minutes_14d_a": _fatigue_minutes(rec_a, pid_a, days=14) if pid_a and not rec_a.empty else 0.0,
        "fatigue_minutes_14d_b": _fatigue_minutes(rec_b, pid_b, days=14) if pid_b and not rec_b.empty else 0.0,
        "rest_days_a": 7.0,
        "rest_days_b": 7.0,
        "travel_km_a": travel_km(last_t_a, tourney_name) or 0.0,
        "travel_km_b": travel_km(last_t_b, tourney_name) or 0.0,
        "tz_shift_a": timezone_shift_hours(last_t_a, tourney_name) or 0.0,
        "tz_shift_b": timezone_shift_hours(last_t_b, tourney_name) or 0.0,
        "surface_wr_a": _surface_wr(rec_a, pid_a, surface) if pid_a and not rec_a.empty else 0.5,
        "surface_wr_b": _surface_wr(rec_b, pid_b, surface) if pid_b and not rec_b.empty else 0.5,
        "hold_pct_a": hold_a,
        "hold_pct_b": hold_b,
        "break_pct_a": break_a,
        "break_pct_b": break_b,
        "hold_break_edge": hold_a - break_b,
        "form_wr_10_a": _surface_wr(rec_a, pid_a, surface) if pid_a else 0.5,
        "form_wr_10_b": _surface_wr(rec_b, pid_b, surface) if pid_b else 0.5,
        "cpi_norm": cpi if cpi is not None else 1.0,
        "level_weight": 1.0,
        "best_of": best_of,
        "is_bo5": int(best_of >= 5),
    }

    tour = "w" if "wta" in str(tourney_name or "").lower() or "women" in str(tourney_name or "").lower() else "m"
    feat["_pressure_a"] = enrich_pressure_profile(
        player_pressure_profile(player_a, tour=tour), pid_a, m
    )
    feat["_pressure_b"] = enrich_pressure_profile(
        player_pressure_profile(player_b, tour=tour), pid_b, m
    )
    feat["n_dataset_a"] = float(dataset_match_count(pid_a, m))
    feat["n_dataset_b"] = float(dataset_match_count(pid_b, m))
    out = {k: float(feat.get(k, 0.0) or 0.0) for k in FEATURE_COLS}
    out["_pressure_a"] = feat.get("_pressure_a")
    out["_pressure_b"] = feat.get("_pressure_b")
    return out
