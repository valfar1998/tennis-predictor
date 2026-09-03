"""Shrinkage bayesiano modello→mercato e filtri sharp consensus / ITF."""

from __future__ import annotations

from typing import Any

from modules.advisor.itf_governance import effective_itf_params, itf_quality_reasons
from modules.advisor.risk_controls import infer_tourney_level
from modules.advisor.value import devig_multiplicative, devig_power, devig_shin
from modules.constants import (
    BAYES_SHRINK_MIN_MATCHES,
    BAYES_SHRINK_W_ITF,
    ITF_BET_FREEZE,
    MKT_DIVERGENCE_MAX,
    MKT_DIVERGENCE_SOFT,
    SHARP_HIGH_ODDS_MIN,
    SHARP_ODDS_SOURCES,
    TOURNEY_LEVEL_CODE,
)

_DEVIG = {
    "shin": devig_shin,
    "power": devig_power,
    "multiplicative": devig_multiplicative,
}


def tourney_level_numeric(level: str | None = None, tourney: str | None = None) -> float:
    code = infer_tourney_level(tourney, level)
    return float(TOURNEY_LEVEL_CODE.get(code, 2.0))


def _data_density_min(prediction: dict[str, Any]) -> int:
    dd = prediction.get("data_density")
    if isinstance(dd, dict):
        return int(dd.get("min") or min(int(dd.get("a") or 0), int(dd.get("b") or 0)))
    if dd is not None:
        return int(dd)
    return int(prediction.get("data_density_min") or 0)


def shrink_model_weight(
    level: str,
    data_density_min: int,
    *,
    mkt_divergence: float = 0.0,
) -> float:
    """Peso modello w in P_adj = w·P_model + (1-w)·P_mkt.

    Con alta incertezza o divergenza dal mercato, w scende (prior di mercato più forte).
    """
    if level == "S" or data_density_min < BAYES_SHRINK_MIN_MATCHES:
        w = float(effective_itf_params().get("shrink_w_itf", BAYES_SHRINK_W_ITF))
    else:
        w_itf = effective_itf_params().get("shrink_w_itf", BAYES_SHRINK_W_ITF)
        by_level = {"G": 0.75, "M": 0.70, "F": 0.68, "A": 0.58, "C": 0.35, "S": w_itf}
        base = by_level.get(level, 0.55)
        if data_density_min < 25:
            w = min(base, 0.32)
        elif data_density_min < 50:
            w = min(base, 0.45)
        else:
            w = base

    # Degrado confidenza se il modello diverge dal mercato
    div = abs(float(mkt_divergence or 0.0))
    if div >= MKT_DIVERGENCE_MAX:
        w *= 0.25
    elif div >= MKT_DIVERGENCE_SOFT:
        # lineare: a SOFT → ×0.70, a MAX → ×0.25
        span = max(MKT_DIVERGENCE_MAX - MKT_DIVERGENCE_SOFT, 1e-6)
        t = (div - MKT_DIVERGENCE_SOFT) / span
        w *= 0.70 - 0.45 * t
    return float(max(0.05, min(0.90, w)))


def market_probs(
    odds_a: float,
    odds_b: float,
    *,
    method: str = "shin",
) -> tuple[float, float]:
    fn = _DEVIG.get(method, devig_shin)
    return fn(odds_a, odds_b)


def apply_bayesian_shrinkage(
    prediction: dict[str, Any],
    odds_a: float | None,
    odds_b: float | None,
    *,
    method: str = "shin",
) -> dict[str, Any]:
    """Shrink P_win_a verso probabilità di mercato de-vigged."""
    out = dict(prediction)
    if not odds_a or not odds_b or float(odds_a) <= 1.01 or float(odds_b) <= 1.01:
        return out

    p_model = float(prediction.get("p_win_a_raw") or prediction.get("p_win_a") or 0.5)
    if prediction.get("p_win_a_raw") is None:
        out["p_win_a_raw"] = round(p_model, 4)

    mkt_a, _ = market_probs(float(odds_a), float(odds_b), method=method)
    level = infer_tourney_level(out.get("tourney"), out.get("tourney_level"))
    density = _data_density_min(out)
    div = abs(p_model - mkt_a)
    w = shrink_model_weight(level, density, mkt_divergence=div)
    p_adj = w * p_model + (1.0 - w) * mkt_a

    out["p_win_a"] = round(p_adj, 4)
    out["market_shrinkage"] = {
        "w": round(w, 4),
        "p_model": round(p_model, 4),
        "p_mkt_a": round(mkt_a, 4),
        "data_density_min": density,
        "tourney_level": level,
        "divergence_degrade": div >= MKT_DIVERGENCE_SOFT,
    }
    out["mkt_divergence"] = round(div, 4)
    out["tourney_level_code"] = tourney_level_numeric(level, out.get("tourney"))
    return out


def itf_freeze_reasons(prediction: dict[str, Any]) -> list[str]:
    """Deprecato: freeze totale disabilitato; usa itf_quality_reasons."""
    if not ITF_BET_FREEZE:
        return []
    level = infer_tourney_level(prediction.get("tourney"), prediction.get("tourney_level"))
    tourney = str(prediction.get("tourney") or "").lower()
    if level == "S" or "itf" in tourney:
        return ["ITF freeze: tornei minori sospesi fino a ricalibrazione del modello"]
    return []


def itf_gate_reasons(prediction: dict[str, Any]) -> list[str]:
    return itf_freeze_reasons(prediction) + itf_quality_reasons(prediction)


def sharp_consensus_reasons(pick: dict[str, Any], prediction: dict[str, Any]) -> list[str]:
    """Quote > 3.0 su soft book richiedono conferma Betfair/Pinnacle."""
    odds = float(pick.get("odds") or 0)
    if odds <= SHARP_HIGH_ODDS_MIN:
        return []

    src = str(pick.get("odds_source") or prediction.get("odds_source") or "").lower()
    if src in SHARP_ODDS_SOURCES:
        return []

    close = prediction.get("close_odds") or prediction.get("pinnacle_odds") or {}
    close_src = str(close.get("source") or prediction.get("close_source") or "").lower()
    if not any(tag in close_src for tag in SHARP_ODDS_SOURCES):
        return [
            f"sharp consensus: quota {odds:.2f} > {SHARP_HIGH_ODDS_MIN:.0f} "
            "senza quote Betfair/Pinnacle verificate"
        ]

    side = str(pick.get("side") or "")
    sharp_odd = close.get("a") if side == "A" else close.get("b")
    if not sharp_odd or float(sharp_odd) <= 1.01:
        return ["sharp consensus: quota sharp assente sul pick"]

    sharp_odd_f = float(sharp_odd)
    if sharp_odd_f < odds * 0.55:
        return [
            f"sharp consensus: Betfair {sharp_odd_f:.2f} vs {odds:.2f} "
            "— mercato sharp non conferma lo sfavorito"
        ]

    pick_prob = float(pick.get("probability") or 0)
    sharp_implied = 1.0 / sharp_odd_f
    if pick_prob < sharp_implied * 0.85:
        return [
            f"sharp consensus: P {pick_prob:.0%} < implicita sharp {sharp_implied:.0%}"
        ]
    return []


def model_market_divergence_reasons(
    pick: dict[str, Any],
    prediction: dict[str, Any],
    *,
    max_divergence: float = MKT_DIVERGENCE_MAX,
) -> list[str]:
    # Preferisci divergenza pre-shrink (p_model vs mkt) se disponibile
    shrink = prediction.get("market_shrinkage") or {}
    if shrink.get("p_model") is not None and shrink.get("p_mkt_a") is not None:
        side = str(pick.get("side") or "A")
        p_model = float(shrink["p_model"])
        p_mkt = float(shrink["p_mkt_a"])
        if side == "B":
            p_model = 1.0 - p_model
            p_mkt = 1.0 - p_mkt
    else:
        p_model = pick.get("probability")
        p_mkt = pick.get("mkt_prob")
        if p_model is None or p_mkt is None:
            return []
        p_model, p_mkt = float(p_model), float(p_mkt)

    div = abs(p_model - p_mkt)
    if div > max_divergence:
        return [
            f"sanity: divergenza modello/mercato {div:.0%} > {max_divergence:.0%}"
        ]
    return []
