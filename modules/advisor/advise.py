"""Advisory layer: value bet recommendations."""

from __future__ import annotations

from modules.advisor.staking import fractional_kelly, model_uncertainty_reasons, no_bet_reasons
from modules.advisor.value import enrich_value
from modules.constants import MIN_EDGE


def advise(
    prediction: dict,
    odds_a: float | None,
    odds_b: float | None,
    *,
    source: str = "book",
    devig_method: str = "shin",
    min_edge: float = MIN_EDGE,
) -> dict:
    enriched = enrich_value(prediction, odds_a, odds_b, source=source, method=devig_method)
    value = enriched.get("value")
    if not value:
        enriched["action"] = "no_bet"
        enriched["reason"] = "quote assenti"
        return enriched

    plays = []
    uncertainty = model_uncertainty_reasons(enriched)
    for pick in value["picks"]:
        reasons = no_bet_reasons(pick, min_edge=min_edge) + uncertainty
        kelly = fractional_kelly(pick["probability"], pick["odds"]) if not reasons else 0.0
        plays.append({
            **pick,
            "kelly": round(kelly, 4),
            "action": "bet" if not reasons else "no_bet",
            "no_bet_reasons": reasons,
        })

    best_play = max(plays, key=lambda x: x.get("ev", -1))
    enriched["plays"] = plays
    enriched["action"] = best_play["action"]
    enriched["recommended"] = best_play if best_play["action"] == "bet" else None

    if enriched.get("recommended"):
        from modules.advisor.pinnacle_clv import clv_vs_pinnacle

        ps = enriched.get("pinnacle_odds") or {}
        side = enriched["recommended"].get("side", "A")
        clv_info = clv_vs_pinnacle(
            pick_side=side,
            odds_bet=float(enriched["recommended"]["odds"]),
            ps_a=ps.get("a"),
            ps_b=ps.get("b"),
        )
        enriched["recommended"].update({k: v for k, v in clv_info.items() if v is not None})
        enriched["clv"] = clv_info.get("clv")
        enriched["beat_close"] = clv_info.get("beat_close")

    return enriched
