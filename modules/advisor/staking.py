"""Kelly frazionato con cap per tennis (2-way moneyline)."""

from __future__ import annotations

from typing import Any

from modules.constants import (
    KELLY_CAP,
    KELLY_CAP_BY_LEVEL,
    KELLY_FRACTION,
    MIN_EDGE,
    MIN_PROB_PLAY,
    ODDS_VARIANCE_REF,
)


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


def odds_sharpe(prob: float, odds: float, *, ref: float = ODDS_VARIANCE_REF) -> float:
    """Score tipo Sharpe: Kelly-unit × sostenibilità².

    Preferisce sempre quote corte a parità di edge unitario; una 4.90 con P gonfiata
    resta sotto una 1.80@60% tipica.
    """
    if odds <= 1.01 or prob <= 0:
        return -1.0
    ev = prob * odds - 1.0
    if ev <= 0:
        return -1.0
    kelly_unit = ev / max(odds - 1.0, 1e-6)
    sustain = min(1.0, ref / max(odds, 1.01))
    return float(kelly_unit * (sustain ** 2))


def kelly_adjusted_rank(prob: float, odds: float, *, kelly: float | None = None) -> float:
    """Chiave di ranking: Kelly × fattore sostenibilità quota."""
    k = float(kelly) if kelly is not None else fractional_kelly(prob, odds)
    if k <= 0 or odds <= 1.01:
        return -1.0
    # Penalità soft: a odds=ref → 1.0; a odds=5 → ~0.45
    sustain = min(1.0, ODDS_VARIANCE_REF / max(odds, 1.01))
    sustain = 0.35 + 0.65 * sustain
    return float(k * sustain)


def kelly_cap_for_level(level: str | None = None, tourney: str | None = None) -> float:
    """Cap Kelly per livello torneo (Grand Slam vs Challenger/ITF)."""
    from modules.advisor.risk_controls import infer_tourney_level

    code = infer_tourney_level(tourney, level)
    return float(KELLY_CAP_BY_LEVEL.get(code, KELLY_CAP_BY_LEVEL.get("A", KELLY_CAP)))


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


def model_uncertainty_reasons(prediction: dict) -> list[str]:
    reasons: list[str] = []
    if prediction.get("model_low_confidence"):
        reasons.append("modello incerto: uno o entrambi i giocatori non identificati nel database")
    p = prediction.get("p_win_a")
    p_elo = prediction.get("p_elo")
    if (
        p is not None
        and p_elo is not None
        and prediction.get("p_ml") is None
        and abs(float(p) - 0.5) < 0.04
        and abs(float(p_elo) - 0.5) < 0.04
    ):
        reasons.append("modello ~50/50 senza ML: probabile artefatto, non edge reale")
    from modules.advisor.risk_controls import infer_tourney_level

    level = infer_tourney_level(prediction.get("tourney"), prediction.get("tourney_level"))
    if (
        level == "S"
        and p is not None
        and abs(float(p) - 0.5) < 0.06
    ):
        reasons.append("modello ~50/50 su ITF: copertura dati insufficiente")
    return reasons


def apply_retirement_filter(
    play: dict,
    *,
    bookmaker: str = "default",
    player_injury_risk: float = 0.0,
    **kwargs,
) -> dict:
    """Regola EV/stake in base alla policy ritiro del bookmaker e P(ritiro)."""
    from modules.advisor.retirement_risk import adjust_play_for_retirement

    p_retire = float(player_injury_risk or play.get("p_retire") or 0.0)
    return adjust_play_for_retirement(play, p_retire=p_retire, bookmaker=bookmaker)
