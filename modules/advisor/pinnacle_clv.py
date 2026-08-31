"""CLV standardizzato vs chiusura Pinnacle (Shin de-vig)."""

from __future__ import annotations

from modules.advisor.staking import beat_close, clv_prob
from modules.advisor.value import devig_shin


def pinnacle_fair_probs(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Probabilità fair da quote Pinnacle PSW/PSL."""
    return devig_shin(float(odds_a), float(odds_b))


def clv_vs_pinnacle(
    *,
    pick_side: str,
    odds_bet: float,
    ps_a: float | None,
    ps_b: float | None,
) -> dict:
    """CLV del pick vs chiusura Pinnacle (fair prob e beat close)."""
    if not ps_a or not ps_b or ps_a <= 1.01 or ps_b <= 1.01:
        return {"clv": None, "beat_close": None, "pinnacle_fair": None}
    fair_a, fair_b = pinnacle_fair_probs(ps_a, ps_b)
    close_odds = ps_a if pick_side == "A" else ps_b
    clv = clv_prob(odds_bet, close_odds)
    return {
        "clv": clv,
        "beat_close": beat_close(odds_bet, close_odds),
        "pinnacle_fair": round(fair_a if pick_side == "A" else fair_b, 4),
        "ps_close_a": ps_a,
        "ps_close_b": ps_b,
    }
