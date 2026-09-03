"""Advisory layer: value bet recommendations."""

from __future__ import annotations

from modules.advisor.itf_governance import ev_sanity_cap_high_for
from modules.advisor.market_calibration import (
    apply_bayesian_shrinkage,
    itf_gate_reasons,
    model_market_divergence_reasons,
    sharp_consensus_reasons,
)
from modules.advisor.staking import (
    apply_retirement_filter,
    fractional_kelly,
    kelly_adjusted_rank,
    kelly_cap_for_level,
    model_uncertainty_reasons,
    no_bet_reasons,
    odds_sharpe,
)
from modules.advisor.value import compute_ev, enrich_value
from modules.constants import (
    EV_REVIEW_THRESHOLD,
    EV_SANITY_CAP_LOW_ODDS,
    EV_SANITY_MAX_ODDS,
    MIN_EDGE,
    SHARP_HIGH_ODDS_MIN,
)


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
    """Blocca EV implausibile o divergenza modello/mercato eccessiva."""
    ev = pick.get("ev")
    odds = pick.get("odds")
    if ev is None or odds is None:
        return []
    if prediction.get("manual_override"):
        return []

    odds_f = float(odds)
    ev_f = float(ev)
    cap_high = ev_sanity_cap_high_for(prediction)
    max_allowed_ev = cap_high if odds_f > SHARP_HIGH_ODDS_MIN else EV_SANITY_CAP_LOW_ODDS

    reasons: list[str] = []
    if ev_f > max_allowed_ev:
        reasons.append(
            f"sanity: EV {ev_f:+.1%} su quota {odds_f:.2f} "
            f"(>{max_allowed_ev:.0%}) — edge irrealistico, scartato"
        )
    # Legacy inversion check su quote basse
    if ev_f > EV_SANITY_CAP_LOW_ODDS and odds_f < EV_SANITY_MAX_ODDS:
        reasons.append(
            f"sanity: EV {ev_f:+.1%} su quota {odds_f:.2f} "
            f"(>{EV_SANITY_CAP_LOW_ODDS:.0%} e <{EV_SANITY_MAX_ODDS:.2f}) "
            "— possibile inversione/dato errato"
        )
    reasons.extend(model_market_divergence_reasons(pick, prediction))
    return reasons


def needs_ev_review(pick: dict, prediction: dict) -> bool:
    """EV alto ma sotto hard-cap → revisione manuale (no alert automatico)."""
    if prediction.get("manual_override"):
        return False
    ev = pick.get("ev")
    odds = pick.get("odds")
    if ev is None or odds is None:
        return False
    ev_f = float(ev)
    if ev_f <= EV_REVIEW_THRESHOLD:
        return False
    odds_f = float(odds)
    cap_high = ev_sanity_cap_high_for(prediction)
    max_allowed = cap_high if odds_f > SHARP_HIGH_ODDS_MIN else EV_SANITY_CAP_LOW_ODDS
    return ev_f <= max_allowed


def _rank_key(play: dict) -> tuple:
    """Ordina per Kelly-adjusted / Sharpe, non EV grezzo."""
    action_rank = {"bet": 2, "review": 1, "no_bet": 0}.get(str(play.get("action")), 0)
    return (
        action_rank,
        float(play.get("kelly_adj_rank") or -1),
        float(play.get("odds_sharpe") or -1),
        float(play.get("kelly") or -1),
    )


def advise(
    prediction: dict,
    odds_a: float | None,
    odds_b: float | None,
    *,
    source: str = "book",
    devig_method: str = "shin",
    min_edge: float = MIN_EDGE,
    dropping_row: dict | None = None,
    bookmaker: str = "default",
    retirement_context: dict | None = None,
) -> dict:
    """Calcola value bet usando quote correnti e shrinkage bayesiano."""
    shrunk = apply_bayesian_shrinkage(
        prediction,
        odds_a,
        odds_b,
        method=devig_method,
    )
    enriched = enrich_value(shrunk, odds_a, odds_b, source=source, method=devig_method)
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
    uncertainty = (
        model_uncertainty_reasons(enriched)
        + itf_gate_reasons(enriched)
    )
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
            + sharp_consensus_reasons(pick, enriched)
        )
        review = False
        if not reasons and needs_ev_review(pick, enriched):
            review = True
            reasons = [
                f"review: EV {float(pick['ev']):+.1%} > {EV_REVIEW_THRESHOLD:.0%} "
                "— edge raro, verifica manuale (no alert auto)"
            ]

        kelly_info = fractional_kelly(pick["probability"], pick["odds"], cap=kelly_cap)
        kelly = kelly_info if not reasons or review else 0.0
        sharpe = odds_sharpe(pick["probability"], pick["odds"])
        adj_rank = kelly_adjusted_rank(pick["probability"], pick["odds"], kelly=kelly_info)
        if review:
            action = "review"
        elif not reasons:
            action = "bet"
        else:
            action = "no_bet"

        play_row = {
            **pick,
            "kelly": round(kelly, 4),
            "kelly_info": round(kelly_info, 4),
            "kelly_cap": kelly_cap,
            "odds_sharpe": round(sharpe, 4),
            "kelly_adj_rank": round(adj_rank, 6),
            "action": action,
            "no_bet_reasons": reasons if action == "no_bet" else ([] if action == "bet" else reasons),
            "review_reasons": reasons if action == "review" else [],
        }
        if retirement_context and action in ("bet", "review"):
            play_row = apply_retirement_filter(
                play_row,
                bookmaker=bookmaker,
                **retirement_context,
            )
            if play_row.get("kelly_adj") is not None:
                play_row["kelly"] = play_row["kelly_adj"]
                play_row["kelly_adj_rank"] = round(
                    kelly_adjusted_rank(
                        play_row["probability"],
                        play_row["odds"],
                        kelly=play_row["kelly"],
                    ),
                    6,
                )
        plays.append(play_row)

    best_play = max(plays, key=_rank_key)
    best_play = dict(best_play)
    best_play["kelly_cap"] = kelly_cap
    enriched["plays"] = plays
    enriched["best_play"] = best_play
    enriched["action"] = best_play["action"]
    # recommended = solo bet/review (alert/staking). best_play resta la previsione.
    enriched["recommended"] = best_play if best_play["action"] in ("bet", "review") else None
    if prediction.get("tourney_level"):
        enriched["tourney_level"] = prediction["tourney_level"]
    return enriched


def display_pick(pred: dict) -> dict:
    """Lato da mostrare in tabella: bet/review, altrimenti best_play / value.best."""
    rec = pred.get("recommended") or pred.get("best_play")
    if rec:
        return rec
    plays = pred.get("plays") or []
    if plays:
        return max(plays, key=_rank_key)
    return (pred.get("value") or {}).get("best") or {}
