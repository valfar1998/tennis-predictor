"""CLV live a cascata: Betfair LTP (primario) → OddsPortal → tennis-data storico."""

from __future__ import annotations

from modules.advisor.pinnacle_clv import clv_vs_pinnacle, pinnacle_fair_probs
from modules.advisor.staking import beat_close, clv_prob


def betfair_fair_probs(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Proxy chiusura Pinnacle via Betfair Exchange (Shin de-vig, r≈0.98 su ATP/WTA)."""
    return pinnacle_fair_probs(odds_a, odds_b)


def clv_vs_close(
    *,
    pick_side: str,
    odds_bet: float,
    close_a: float | None,
    close_b: float | None,
    source: str = "close",
) -> dict:
    """CLV vs quote di chiusura (qualsiasi fonte)."""
    if not close_a or not close_b or close_a <= 1.01 or close_b <= 1.01:
        return {"clv": None, "beat_close": None, "close_fair": None, "close_source": source}
    close_odds = close_a if pick_side == "A" else close_b
    fair_a, fair_b = pinnacle_fair_probs(close_a, close_b)
    return {
        "clv": clv_prob(odds_bet, close_odds),
        "beat_close": beat_close(odds_bet, close_odds),
        "close_fair": round(fair_a if pick_side == "A" else fair_b, 4),
        "close_source": source,
        "close_a": close_a,
        "close_b": close_b,
    }


def resolve_close_odds(
    player_a: str,
    player_b: str,
    *,
    date: str | None = None,
    tour: str = "ATP",
    betfair_event_id: str | None = None,
    betfair_odds: dict | None = None,
) -> dict | None:
    """Risolve quote di chiusura con cascade a costo zero.

    1. Betfair LTP/BSP (fonte primaria — stessa del betting)
    2. OddsPortal cache (post-match batch)
    3. tennis-data.co.uk cache (match già giocati)
    """
    # 1) Betfair LTP (live o cache aggiornata)
    try:
        from modules.data_update.betfair import lookup_betfair_close

        bf = lookup_betfair_close(
            player_a,
            player_b,
            event_id=betfair_event_id,
            match_date=date,
            bet_odds=betfair_odds,
        )
        if bf:
            return bf
    except Exception:
        pass

    # 2) OddsPortal cache (post-match batch)
    try:
        from modules.data_update.oddsportal_close import lookup_oddsportal_close

        op = lookup_oddsportal_close(player_a, player_b, date=date)
        if op:
            return op
    except Exception:
        pass

    # 3) tennis-data storico
    try:
        from modules.data_update.tennis_data_portal import lookup_pinnacle_odds

        td = lookup_pinnacle_odds(player_a, player_b, date=date, tour=tour)
        if td:
            return {**td, "source": td.get("source") or "tennis-data.co.uk"}
    except Exception:
        pass

    return None


def enrich_clv(prediction: dict, *, pick_side: str, odds_bet: float) -> dict:
    """Aggiunge CLV al pick usando close odds già risolte o cascade."""
    close = prediction.get("close_odds") or prediction.get("pinnacle_odds") or {}
    if not close.get("a") or not close.get("b"):
        close = resolve_close_odds(
            str(prediction.get("player_a") or ""),
            str(prediction.get("player_b") or ""),
            date=str(prediction.get("date") or "")[:10],
            tour=str(prediction.get("tour") or "ATP"),
            betfair_event_id=prediction.get("betfair_event_id"),
            betfair_odds=prediction.get("book_odds"),
        ) or {}

    info = clv_vs_close(
        pick_side=pick_side,
        odds_bet=odds_bet,
        close_a=close.get("a"),
        close_b=close.get("b"),
        source=str(close.get("source") or "unknown"),
    )
    # Backward compat con pinnacle_clv
    if info.get("clv") is not None:
        info["pinnacle_fair"] = info.get("close_fair")
        info["ps_close_a"] = close.get("a")
        info["ps_close_b"] = close.get("b")
    return info
