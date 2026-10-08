"""Policy campione ROI Telegram — solo tip allineate al metodo post-calibrazione.

Esclude le tip underdog con edge grezzo vs mercato ≤ noise floor (false-edge storiche).
Non cancella history: filtra solo il calcolo ROI / KPI.
"""

from __future__ import annotations

from typing import Any

from modules.constants import (
    MAX_ODDS_PLAY,
    MIN_EDGE,
    MIN_KELLY,
    MIN_ODDS_PLAY,
    MIN_PROB_PLAY,
    UNDERDOG_EDGE_NOISE_FLOOR,
    UNDERDOG_MKT_MAX,
)

# Deploy calibrazione anti-overconfidence underdog
ROI_POLICY_SINCE = "2026-10-08"
ROI_POLICY_LABEL = "post-calibrazione underdog (noise floor 6pp)"


def _f(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        try:
            v = row.get(key)
            if v is not None and float(v) == float(v):
                return float(v)
        except (TypeError, ValueError):
            continue
    return default


def tip_aligns_new_method(row: dict[str, Any]) -> bool:
    """True se la tip avrebbe ancora senso sotto shrink + noise floor underdog."""
    odds = _f(row, "odds_alert", "odds")
    kelly = _f(row, "kelly_alert", "kelly")
    prob = _f(row, "probability")
    if odds < MIN_ODDS_PLAY or odds > MAX_ODDS_PLAY:
        return False
    if kelly < MIN_KELLY:
        return False
    if prob < MIN_PROB_PLAY:
        return False

    # Probabilità implicita grezza (1/odds): aprossimazione senza overround 2-way
    p_mkt = 1.0 / odds
    # Underdog: mercato < 45%
    if p_mkt <= UNDERDOG_MKT_MAX:
        raw_edge = prob - p_mkt
        # Edge sotto noise floor → con il nuovo metodo P torna al mercato → no tip
        if raw_edge <= UNDERDOG_EDGE_NOISE_FLOOR:
            return False
        # Tiene solo eccesso oltre il floor (stima conservativa del post-shrink)
        p_adj = p_mkt + (raw_edge - UNDERDOG_EDGE_NOISE_FLOOR) * 0.45
        ev = p_adj * odds - 1.0
        return ev >= MIN_EDGE

    # Favorito / zona media: EV sulla P archiviata
    ev = prob * odds - 1.0
    return ev >= MIN_EDGE


def filter_roi_tips(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filtra action=bet allineate; ritorna (kept, stats)."""
    bets = [r for r in rows if r.get("action") == "bet"]
    kept = [r for r in bets if tip_aligns_new_method(r)]
    excluded = len(bets) - len(kept)
    return kept, {
        "n_bets_all": len(bets),
        "n_kept": len(kept),
        "n_excluded": excluded,
        "policy_since": ROI_POLICY_SINCE,
        "policy": ROI_POLICY_LABEL,
    }
