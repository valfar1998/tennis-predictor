"""Parametri ITF adattivi in base al BCR settle (no freeze totale)."""

from __future__ import annotations

from typing import Any

from modules.advisor.risk_controls import infer_tourney_level
from modules.constants import (
    BAYES_SHRINK_W_ITF,
    EV_SANITY_CAP_HIGH_ODDS,
    ITF_BCR_MIN_N,
    ITF_BCR_RELAX_THRESHOLD,
    ITF_BCR_STRICT_THRESHOLD,
    ITF_EV_CAP_RELAXED,
    ITF_EV_CAP_STRICT,
    ITF_MIN_DATA_DENSITY_STRICT,
    ITF_SHRINK_W_RELAXED,
)


def is_itf_prediction(prediction: dict[str, Any]) -> bool:
    level = infer_tourney_level(prediction.get("tourney"), prediction.get("tourney_level"))
    tourney = str(prediction.get("tourney") or "").lower()
    return level == "S" or "itf" in tourney


def is_itf_history_row(row: dict[str, Any]) -> bool:
    return is_itf_prediction({"tourney": row.get("tourney"), "tourney_level": row.get("tourney_level")})


def compute_itf_bcr(*, betfair_only: bool = True) -> dict[str, Any]:
    """BCR solo su pick ITF settle con chiusura disponibile."""
    from modules.advisor.live_metrics import compute_bcr

    base = compute_bcr(betfair_only=betfair_only, days=None)
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
    - insufficient: campione < ITF_BCR_MIN_N → baseline
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
        "min_data_density": 0,
        "regime": "default",
        "bcr_itf": bcr_info,
    }

    if n < ITF_BCR_MIN_N or bcr is None:
        params["regime"] = "insufficient_sample"
    elif float(bcr) >= ITF_BCR_RELAX_THRESHOLD:
        params["regime"] = "relaxed"
        params["shrink_w_itf"] = ITF_SHRINK_W_RELAXED
        params["ev_sanity_cap_high"] = ITF_EV_CAP_RELAXED
    elif float(bcr) < ITF_BCR_STRICT_THRESHOLD:
        params["regime"] = "strict"
        params["ev_sanity_cap_high"] = ITF_EV_CAP_STRICT
        params["min_data_density"] = ITF_MIN_DATA_DENSITY_STRICT
    else:
        params["regime"] = "default"

    _CACHE["params"] = params
    return params


_CACHE: dict[str, Any] = {}


def itf_quality_reasons(prediction: dict[str, Any]) -> list[str]:
    """Gate ITF senza freeze totale: densità dati quando BCR è molto basso."""
    if not is_itf_prediction(prediction):
        return []

    params = effective_itf_params()
    min_density = int(params.get("min_data_density") or 0)
    if min_density <= 0:
        return []

    dd = prediction.get("data_density")
    if isinstance(dd, dict):
        density = int(dd.get("min") or min(int(dd.get("a") or 0), int(dd.get("b") or 0)))
    else:
        density = int(dd or prediction.get("data_density_min") or 0)

    if density < min_density:
        bcr_pct = params.get("bcr_itf", {}).get("bcr_pct")
        return [
            f"ITF strict: data_density {density} < {min_density} "
            f"(BCR ITF {bcr_pct}% — regime {params.get('regime')})"
        ]
    return []


def ev_sanity_cap_high_for(prediction: dict[str, Any]) -> float:
    if not is_itf_prediction(prediction):
        return float(EV_SANITY_CAP_HIGH_ODDS)
    return float(effective_itf_params().get("ev_sanity_cap_high") or EV_SANITY_CAP_HIGH_ODDS)
