"""Parametri ITF/Challenger adattivi in base al BCR settle (no freeze totale).

Post–Fase 1: baseline densità più alta su ITF e sanity EV più stretta sui minori;
ATP/Masters restano più flessibili (vedi ``ev_sanity_cap_*_for``).
"""

from __future__ import annotations

from typing import Any

from modules.advisor.risk_controls import infer_tourney_level
from modules.constants import (
    ATP_EV_SANITY_CAP_HIGH_ODDS,
    ATP_EV_SANITY_CAP_LOW_ODDS,
    BAYES_SHRINK_W_ITF,
    CHALLENGER_MIN_DATA_DENSITY,
    EV_SANITY_CAP_HIGH_ODDS,
    EV_SANITY_CAP_LOW_ODDS,
    ITF_BCR_MIN_N,
    ITF_BCR_RELAX_THRESHOLD,
    ITF_BCR_STRICT_THRESHOLD,
    ITF_EV_CAP_LOW_ODDS,
    ITF_EV_CAP_RELAXED,
    ITF_EV_CAP_STRICT,
    ITF_MIN_DATA_DENSITY_BASELINE,
    ITF_MIN_DATA_DENSITY_STRICT,
    ITF_SHRINK_W_RELAXED,
)


def is_itf_prediction(prediction: dict[str, Any]) -> bool:
    level = infer_tourney_level(prediction.get("tourney"), prediction.get("tourney_level"))
    tourney = str(prediction.get("tourney") or "").lower()
    return level == "S" or "itf" in tourney


def is_challenger_prediction(prediction: dict[str, Any]) -> bool:
    level = infer_tourney_level(prediction.get("tourney"), prediction.get("tourney_level"))
    tourney = str(prediction.get("tourney") or "").lower()
    return level == "C" or "challenger" in tourney


def is_itf_history_row(row: dict[str, Any]) -> bool:
    return is_itf_prediction({"tourney": row.get("tourney"), "tourney_level": row.get("tourney_level")})


def _data_density_min(prediction: dict[str, Any]) -> int:
    dd = prediction.get("data_density")
    if isinstance(dd, dict):
        return int(dd.get("min") or min(int(dd.get("a") or 0), int(dd.get("b") or 0)))
    return int(dd or prediction.get("data_density_min") or 0)


def compute_itf_bcr(*, betfair_only: bool = True) -> dict[str, Any]:
    """BCR solo su pick ITF settle con chiusura disponibile."""
    from modules.advisor.live_metrics import compute_bcr

    _ = compute_bcr(betfair_only=betfair_only, days=None)
    from modules.data_update.history import load_history

    rows = load_history(limit=5000)
    settled = [
        r
        for r in rows
        if r.get("hit") is not None
        and r.get("action") == "bet"
        and is_itf_history_row(r)
        and r.get("beat_close") is not None
    ]
    if betfair_only:
        settled = [
            r for r in settled
            if "betfair" in str(r.get("close_source") or "").lower()
        ]

    n = len(settled)
    beats = sum(1 for r in settled if int(r.get("beat_close") or 0) == 1)
    rate = beats / n if n else None

    return {
        "n": n,
        "beats": beats,
        "bcr": round(rate, 4) if rate is not None else None,
        "bcr_pct": round(rate * 100, 1) if rate is not None else None,
        "betfair_only": betfair_only,
        "min_n_for_tuning": ITF_BCR_MIN_N,
        "relax_threshold_pct": round(ITF_BCR_RELAX_THRESHOLD * 100, 1),
        "strict_threshold_pct": round(ITF_BCR_STRICT_THRESHOLD * 100, 1),
    }


def effective_itf_params(*, refresh: bool = False) -> dict[str, Any]:
    """
    Regime:
    - insufficient: campione < ITF_BCR_MIN_N → baseline + densità BASELINE
    - relaxed: BCR >= 20% → più fiducia al modello
    - strict: BCR < 5% → data_density + EV cap ridotto (no freeze)
    - default: tra 5% e 20%
    """
    if not refresh:
        cached = _CACHE.get("params")
        if cached is not None:
            return cached

    bcr_info = compute_itf_bcr(betfair_only=True)
    n = int(bcr_info.get("n") or 0)
    bcr = bcr_info.get("bcr")

    params: dict[str, Any] = {
        "shrink_w_itf": BAYES_SHRINK_W_ITF,
        "ev_sanity_cap_high": EV_SANITY_CAP_HIGH_ODDS,
        "ev_sanity_cap_low": ITF_EV_CAP_LOW_ODDS,
        "min_data_density": ITF_MIN_DATA_DENSITY_BASELINE,
        "regime": "default",
        "bcr_itf": bcr_info,
    }

    if n < ITF_BCR_MIN_N or bcr is None:
        params["regime"] = "insufficient_sample"
        params["min_data_density"] = ITF_MIN_DATA_DENSITY_BASELINE
        params["ev_sanity_cap_high"] = ITF_EV_CAP_STRICT
    elif float(bcr) >= ITF_BCR_RELAX_THRESHOLD:
        params["regime"] = "relaxed"
        params["shrink_w_itf"] = ITF_SHRINK_W_RELAXED
        params["ev_sanity_cap_high"] = ITF_EV_CAP_RELAXED
        params["ev_sanity_cap_low"] = min(ITF_EV_CAP_RELAXED, ITF_EV_CAP_LOW_ODDS + 0.04)
        params["min_data_density"] = max(8, ITF_MIN_DATA_DENSITY_BASELINE - 4)
    elif float(bcr) < ITF_BCR_STRICT_THRESHOLD:
        params["regime"] = "strict"
        params["ev_sanity_cap_high"] = ITF_EV_CAP_STRICT
        params["ev_sanity_cap_low"] = min(ITF_EV_CAP_LOW_ODDS, ITF_EV_CAP_STRICT)
        params["min_data_density"] = ITF_MIN_DATA_DENSITY_STRICT
    else:
        params["regime"] = "default"
        params["ev_sanity_cap_high"] = ITF_EV_CAP_STRICT
        params["min_data_density"] = ITF_MIN_DATA_DENSITY_BASELINE

    _CACHE["params"] = params
    return params


_CACHE: dict[str, Any] = {}


def itf_quality_reasons(prediction: dict[str, Any]) -> list[str]:
    """Gate densità su ITF (sempre) e Challenger (soglia più bassa)."""
    reasons: list[str] = []
    density = _data_density_min(prediction)

    if is_itf_prediction(prediction):
        params = effective_itf_params()
        min_density = int(params.get("min_data_density") or ITF_MIN_DATA_DENSITY_BASELINE)
        if density < min_density:
            bcr_pct = params.get("bcr_itf", {}).get("bcr_pct")
            reasons.append(
                f"ITF densità: data_density {density} < {min_density} "
                f"(BCR ITF {bcr_pct}% — regime {params.get('regime')})"
            )
        return reasons

    if is_challenger_prediction(prediction):
        if density < CHALLENGER_MIN_DATA_DENSITY:
            reasons.append(
                f"Challenger densità: data_density {density} < {CHALLENGER_MIN_DATA_DENSITY}"
            )
    return reasons


def ev_sanity_cap_high_for(prediction: dict[str, Any]) -> float:
    """Cap EV hard su quote lunghe: ITF stretto, ATP/Masters più flessibile."""
    if is_itf_prediction(prediction):
        return float(effective_itf_params().get("ev_sanity_cap_high") or ITF_EV_CAP_STRICT)
    level = infer_tourney_level(prediction.get("tourney"), prediction.get("tourney_level"))
    if level in ("G", "M", "F", "A", "D"):
        return float(ATP_EV_SANITY_CAP_HIGH_ODDS)
    if is_challenger_prediction(prediction):
        return float(EV_SANITY_CAP_HIGH_ODDS)
    return float(EV_SANITY_CAP_HIGH_ODDS)


def ev_sanity_cap_low_for(prediction: dict[str, Any]) -> float:
    """Cap EV hard su quote corte: ITF più severo post–Fase 1."""
    if is_itf_prediction(prediction):
        return float(effective_itf_params().get("ev_sanity_cap_low") or ITF_EV_CAP_LOW_ODDS)
    level = infer_tourney_level(prediction.get("tourney"), prediction.get("tourney_level"))
    if level in ("G", "M", "F", "A", "D"):
        return float(ATP_EV_SANITY_CAP_LOW_ODDS)
    return float(EV_SANITY_CAP_LOW_ODDS)
