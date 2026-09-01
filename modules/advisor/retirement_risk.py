"""Sotto-modello probabilità ritiro in-match + modulazione stake Kelly."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from modules.constants import RETIREMENT_RULES


def _player_age_years(dob_yyyymmdd: int | str | None, ref: datetime | None = None) -> float | None:
    if dob_yyyymmdd is None or pd.isna(dob_yyyymmdd):
        return None
    s = str(int(dob_yyyymmdd))
    if len(s) != 8:
        return None
    try:
        born = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None
    ref = ref or datetime.utcnow()
    return (ref - born).days / 365.25


def historical_retirement_rate(matches: pd.DataFrame, player_id: int) -> float:
    """Frazione match con RET/DEF/WO per giocatore."""
    if matches is None or matches.empty or player_id is None:
        return 0.0
    pid = int(player_id)
    sub = matches[(matches["winner_id"] == pid) | (matches["loser_id"] == pid)]
    if sub.empty:
        return 0.0
    flags = sub["score"].astype(str).str.upper()
    bad = flags.str.contains("RET|DEF|W/O|WO", regex=True, na=False)
    return float(bad.sum()) / len(sub)


def estimate_retirement_risk(
    *,
    age: float | None = None,
    fatigue_minutes_72h: float = 0.0,
    rest_days: float = 7.0,
    medical_timeouts_recent: int = 0,
    historical_retire_rate: float = 0.0,
    is_favorite: bool = False,
) -> float:
    """P(ritiro) prima/durante match [0, 1]."""
    p = 0.015 + float(historical_retire_rate) * 0.35

    if age is not None and age >= 32:
        p += min(0.12, (age - 32) * 0.012)
    if fatigue_minutes_72h > 300:
        p += min(0.10, (fatigue_minutes_72h - 300) / 2500)
    if rest_days < 2:
        p += min(0.08, (2.0 - rest_days) * 0.04)
    if medical_timeouts_recent > 0:
        p += min(0.15, medical_timeouts_recent * 0.05)

    # Favorito acciaccato: rischio maggiore se deve chiudere il match
    if is_favorite:
        p *= 1.08

    return round(min(0.38, max(0.0, p)), 4)


def rule_penalty(bookmaker: str) -> float:
    """Penalità EV/stake per regola ritiro bookmaker."""
    rule = RETIREMENT_RULES.get(str(bookmaker or "").lower(), RETIREMENT_RULES["default"])
    return {
        "1_ball": 0.55,
        "1_set": 0.25,
        "full_match": 0.40,
    }.get(rule, 0.30)


def adjust_play_for_retirement(
    play: dict,
    *,
    p_retire: float,
    bookmaker: str = "default",
) -> dict:
    """Modula EV e Kelly in base a P(ritiro) e regola book."""
    out = dict(play)
    pen = rule_penalty(bookmaker)
    risk = float(p_retire) * pen
    ev = float(out.get("ev") or 0)
    out["p_retire"] = round(float(p_retire), 4)
    out["retirement_rule"] = RETIREMENT_RULES.get(bookmaker.lower(), RETIREMENT_RULES["default"])
    out["ev_adj"] = round(ev * (1.0 - risk), 4)
    kelly = float(out.get("kelly") or 0)
    out["kelly_adj"] = round(kelly * (1.0 - risk * 1.15), 4)
    if p_retire > 0.12:
        out["retirement_warning"] = (
            f"Rischio ritiro {p_retire:.0%} — regola {out['retirement_rule']} "
            f"(penalità stake {risk:.0%})"
        )
    return out
