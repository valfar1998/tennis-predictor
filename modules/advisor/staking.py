"""Kelly frazionato con cap per tennis (2-way moneyline)."""

from __future__ import annotations

from typing import Any

from modules.constants import KELLY_CAP, KELLY_FRACTION, MIN_EDGE, MIN_PROB_PLAY


def kelly_full(prob: float, odds: float) -> float:
    if odds <= 1.01 or prob <= 0:
        return 0.0
    edge = prob * odds - 1.0
    if edge <= 0:
        return 0.0
    return edge / (odds - 1.0)


def fractional_kelly(
    prob: float,
    odds: float,
    *,
    fraction: float = KELLY_FRACTION,
    cap: float = KELLY_CAP,
) -> float:
    stake = kelly_full(prob, odds) * fraction
    return float(min(max(stake, 0.0), cap))


def clv_prob(odds_bet: float | None, odds_close: float | None) -> float | None:
    if not odds_bet or not odds_close or odds_bet <= 1.01 or odds_close <= 1.01:
        return None
    return round((1.0 / odds_close) - (1.0 / odds_bet), 4)


def beat_close(odds_bet: float | None, odds_close: float | None) -> bool | None:
    if not odds_bet or not odds_close:
        return None
    return float(odds_bet) > float(odds_close) + 0.005


def no_bet_reasons(play: dict[str, Any], *, min_edge: float = MIN_EDGE) -> list[str]:
    reasons: list[str] = []
    ev = play.get("ev")
    if play.get("odds_real") is False:
        reasons.append("quota non reale: edge non misurabile")
    elif ev is None:
        reasons.append("quota assente")
    elif float(ev) < min_edge:
        reasons.append(f"EV {float(ev):+.1%} sotto soglia {min_edge:.0%}")
    prob = play.get("probability")
    if prob is not None and float(prob) < MIN_PROB_PLAY:
        reasons.append(f"probabilità {float(prob):.0%} sotto minimo {MIN_PROB_PLAY:.0%}")
    return reasons


def apply_retirement_filter(
    play: dict,
    *,
    bookmaker: str = "default",
    player_injury_risk: float = 0.0,
) -> dict:
    """Regola EV in base alla policy ritiro del bookmaker."""
    from modules.constants import RETIREMENT_RULES

    rule = RETIREMENT_RULES.get(bookmaker.lower(), RETIREMENT_RULES["default"])
    play = dict(play)
    if rule == "1_ball" and player_injury_risk > 0.3:
        play["ev_adj"] = play.get("ev", 0) * 0.7
        play["retirement_warning"] = "Rischio ritiro alto con regola 1-ball served"
    elif rule == "full_match":
        play["ev_adj"] = play.get("ev", 0) * 0.85
        play["retirement_warning"] = "Book void su ritiro: sconto EV"
    else:
        play["ev_adj"] = play.get("ev")
    return play
