"""Advisory layer: value bet recommendations."""

from __future__ import annotations

from modules.advisor.staking import fractional_kelly, kelly_cap_for_level, model_uncertainty_reasons, no_bet_reasons
from modules.advisor.value import compute_ev, enrich_value
from modules.constants import EV_SANITY_CAP, EV_SANITY_MAX_ODDS, MIN_EDGE


def _dropping_aligned(pick_side: str, dropping_row: dict) -> bool:
    side = str(dropping_row.get("side") or "")
    return (pick_side == "A" and side == "1") or (pick_side == "B" and side == "2")


def steam_eroded_reasons(
    pick: dict,
    *,
    dropping_row: dict | None,
    min_edge: float = MIN_EDGE,
    erosion_ratio: float = 0.55,
) -> list[str]:
    """Scarta pick se lo steam ha eroso il margine sulla quota corrente."""
    if not dropping_row:
        return []

    pick_side = str(pick.get("side") or "")
    if not _dropping_aligned(pick_side, dropping_row):
        return []

    open_odds = dropping_row.get("open_odds")
    current_odds = dropping_row.get("current_odds") or pick.get("odds")
    prob = float(pick.get("probability") or 0)
    odds = float(pick.get("odds") or 0)

    if not odds or odds <= 1.01:
        return []

    ev_current = compute_ev(prob, odds)
    if ev_current < min_edge:
        return [f"steam: edge {ev_current:+.1%} sotto soglia su quota corrente ({odds:.2f})"]

    if open_odds and float(open_odds) > 1.01:
        ev_open = compute_ev(prob, float(open_odds))
        if ev_open >= min_edge and ev_current < ev_open * erosion_ratio:
            drop = float(dropping_row.get("drop_pct") or 0)
            return [
                f"steam: margine eroso dal dropping ({drop:.0f}%, "
                f"EV {ev_open:+.1%}→{ev_current:+.1%} su quota corrente)"
            ]

    return []


def ev_sanity_reasons(pick: dict, prediction: dict) -> list[str]:
    """Blocca EV implausibile su quota bassa (inversione nomi/quote o data error)."""
    ev = pick.get("ev")
    odds = pick.get("odds")
    if ev is None or odds is None:
        return []
    if prediction.get("manual_override"):
        return []
    if float(ev) > EV_SANITY_CAP and float(odds) < EV_SANITY_MAX_ODDS:
        return [
            f"sanity: EV {float(ev):+.1%} su quota {float(odds):.2f} "
            f"(>{EV_SANITY_CAP:.0%} e <{EV_SANITY_MAX_ODDS:.2f}) — possibile inversione/dato errato"
        ]
    return []


def advise(
    prediction: dict,
    odds_a: float | None,
    odds_b: float | None,
    *,
    source: str = "book",
    devig_method: str = "shin",
    min_edge: float = MIN_EDGE,
    dropping_row: dict | None = None,
) -> dict:
    """Calcola value bet usando sempre le quote correnti (odds_a/b)."""
    enriched = enrich_value(prediction, odds_a, odds_b, source=source, method=devig_method)
    if prediction.get("pinnacle_odds"):
        enriched["pinnacle_odds"] = prediction["pinnacle_odds"]
    if dropping_row:
        enriched["dropping_odds"] = dropping_row
    value = enriched.get("value")
    if not value:
        enriched["action"] = "no_bet"
        enriched["reason"] = "quote assenti"
        return enriched

    plays = []
    uncertainty = model_uncertainty_reasons(enriched)
    kelly_cap = kelly_cap_for_level(
        prediction.get("tourney_level"),
        prediction.get("tourney"),
    )
    for pick in value["picks"]:
        reasons = (
            no_bet_reasons(pick, min_edge=min_edge)
            + uncertainty
            + steam_eroded_reasons(pick, dropping_row=dropping_row, min_edge=min_edge)
            + ev_sanity_reasons(pick, enriched)
        )
        kelly = (
            fractional_kelly(pick["probability"], pick["odds"], cap=kelly_cap)
            if not reasons
            else 0.0
        )
        plays.append({
            **pick,
            "kelly": round(kelly, 4),
            "kelly_cap": kelly_cap,
            "action": "bet" if not reasons else "no_bet",
            "no_bet_reasons": reasons,
        })

    best_play = max(plays, key=lambda x: x.get("ev", -1))
    enriched["plays"] = plays
    enriched["action"] = best_play["action"]
    enriched["recommended"] = best_play if best_play["action"] == "bet" else None

    if enriched.get("recommended"):
        enriched["recommended"]["kelly_cap"] = kelly_cap
        if prediction.get("tourney_level"):
            enriched["tourney_level"] = prediction["tourney_level"]
        from modules.advisor.clv_live import enrich_clv

        side = enriched["recommended"].get("side", "A")
        clv_info = enrich_clv(
            enriched,
            pick_side=side,
            odds_bet=float(enriched["recommended"]["odds"]),
        )
        enriched["recommended"].update({k: v for k, v in clv_info.items() if v is not None})
        enriched["clv"] = clv_info.get("clv")
        enriched["beat_close"] = clv_info.get("beat_close")
        if clv_info.get("close_source"):
            enriched["close_source"] = clv_info["close_source"]

    return enriched
