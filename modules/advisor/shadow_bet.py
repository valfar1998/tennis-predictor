"""Shadow bet: archivia pick Betfair a Kelly=0 per costruire sample BCR senza bankroll.

Attivo sotto validation freeze e/o circuit breaker: sblocca il deadlock
«nessun bet → BCR bloccato a n≈0 → freeze eterno».
"""

from __future__ import annotations

from typing import Any

from modules.constants import (
    LOWER_TIER_PARTIAL_RESOLVE_OK,
    MIN_PROB_PLAY,
    SHADOW_BET_ENABLED,
    SHADOW_KELLY,
    SHADOW_MIN_EDGE,
)

# Motivi hard: non promuovere a shadow anche se EV è sopra SHADOW_MIN_EDGE
_HARD_REASON_NEEDLES = (
    "sanity:",
    "steam:",
    "divergenza",
    "probabilità",
    "quota non reale",
    "entrambi i giocatori",
    "sharp consensus",
)


def _is_soft_ev_reason(reason: str) -> bool:
    r = str(reason or "").lower()
    return r.startswith("ev ") and "sotto soglia" in r


def _is_soft_confidence_reason(reason: str) -> bool:
    r = str(reason or "").lower()
    # Partial resolve su sharp: non hard per shadow
    return "modello incerto" in r and "entrambi" not in r


def _has_hard_blocker(reasons: list[Any]) -> bool:
    for raw in reasons or []:
        if _is_soft_ev_reason(str(raw)):
            continue
        if LOWER_TIER_PARTIAL_RESOLVE_OK and _is_soft_confidence_reason(str(raw)):
            continue
        rs = str(raw).lower()
        if any(n in rs for n in _HARD_REASON_NEEDLES):
            return True
    return False


def stress_sample_active(*, freeze_active: bool = False, circuit_breaker_active: bool = False) -> bool:
    return bool(freeze_active or circuit_breaker_active)


def maybe_promote_shadow(
    pred: dict[str, Any],
    *,
    freeze_active: bool = False,
    circuit_breaker_active: bool = False,
    enabled: bool = SHADOW_BET_ENABLED,
) -> dict[str, Any] | None:
    """Se eleggibile, restituisce pred con action=shadow (Kelly=0), altrimenti None."""
    if not enabled or not stress_sample_active(
        freeze_active=freeze_active,
        circuit_breaker_active=circuit_breaker_active,
    ):
        return None
    if pred.get("action") in ("bet", "review", "shadow"):
        return None

    src = str(pred.get("odds_source") or "").lower()
    if "betfair" not in src:
        return None
    if not pred.get("betfair_market_id"):
        return None

    pr = pred.get("players_resolved") or {}
    if pred.get("model_low_confidence") and not (pr.get("a") or pr.get("b")):
        return None

    bp = dict(pred.get("best_play") or {})
    if not bp:
        return None
    try:
        ev = float(bp.get("ev"))
        prob = float(bp.get("probability") or 0)
        odds = float(bp.get("odds") or 0)
    except (TypeError, ValueError):
        return None
    if ev < SHADOW_MIN_EDGE or prob < MIN_PROB_PLAY or odds <= 1.01:
        return None
    if bp.get("odds_real") is False:
        return None
    if _has_hard_blocker(list(bp.get("no_bet_reasons") or [])):
        return None

    rec = dict(bp)
    rec["action"] = "shadow"
    rec["kelly"] = SHADOW_KELLY
    rec["kelly_info"] = float(bp.get("kelly_info") or bp.get("kelly") or 0)
    rec["shadow"] = True
    rec["no_bet_reasons"] = []
    rec["shadow_reasons"] = [
        f"shadow BCR: EV {ev:+.1%} ≥ {SHADOW_MIN_EDGE:.0%} "
        f"(freeze/CB — stake {SHADOW_KELLY:.0%})"
    ]

    out = dict(pred)
    out["action"] = "shadow"
    out["shadow"] = True
    out["best_play"] = rec
    out["recommended"] = rec
    return out
