"""Fase 2: abbassamento dinamico e sicuro di MIN_EDGE verso PHASE2_EDGE_FLOOR.

Si attiva solo con campione settle sufficiente e metriche (hit-rate, CLV, BCR)
che giustificano il relax. Non scrive se validation freeze blocca online_learn.
"""

from __future__ import annotations

from typing import Any

from modules.constants import (
    CIRCUIT_BREAKER_MIN_EDGE,
    MIN_EDGE,
    PHASE2_EDGE_BCR_MIN,
    PHASE2_EDGE_BCR_MIN_N,
    PHASE2_EDGE_CLV_MIN,
    PHASE2_EDGE_FLOOR,
    PHASE2_EDGE_HIT_MIN,
    PHASE2_EDGE_RELAX_MIN_N,
    PHASE2_EDGE_STEP,
)


def _avg_clv(rows: list[dict[str, Any]]) -> float | None:
    vals = [float(r["clv"]) for r in rows if r.get("clv") is not None]
    if len(vals) < 8:
        return None
    return round(sum(vals) / len(vals), 5)


def evaluate_dynamic_edge(
    *,
    settled_bets: list[dict[str, Any]] | None = None,
    bcr_betfair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Valuta se e quanto abbassare l'edge minimo.

    Ritorna dict con ``suggested_min_edge``, ``unlocked``, ``reason``, metriche.
    """
    from modules.data_update.history import load_history

    if settled_bets is None:
        rows = load_history(limit=3000)
        settled_bets = [
            r for r in rows if r.get("hit") is not None and r.get("action") == "bet"
        ]

    n = len(settled_bets)
    hits = sum(int(r.get("hit") or 0) for r in settled_bets)
    hit_rate = hits / n if n else None
    avg_clv = _avg_clv(settled_bets)

    if bcr_betfair is None:
        try:
            from modules.advisor.live_metrics import compute_bcr

            bcr_betfair = compute_bcr(betfair_only=True, quality_only=True)
        except Exception:
            bcr_betfair = {}

    bcr_n = int((bcr_betfair or {}).get("n") or 0)
    bcr = (bcr_betfair or {}).get("bcr")

    out: dict[str, Any] = {
        "n_settled": n,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "avg_clv": avg_clv,
        "bcr_n": bcr_n,
        "bcr": bcr,
        "min_n": PHASE2_EDGE_RELAX_MIN_N,
        "floor": PHASE2_EDGE_FLOOR,
        "base": MIN_EDGE,
        "suggested_min_edge": MIN_EDGE,
        "unlocked": False,
        "reason": "insufficient_sample",
    }

    if n < PHASE2_EDGE_RELAX_MIN_N:
        out["reason"] = f"n_settled {n} < {PHASE2_EDGE_RELAX_MIN_N}"
        return out

    # Gate qualità: almeno uno tra CLV medio ≥0 (n≥8) oppure BCR quality ok
    clv_ok = avg_clv is not None and avg_clv >= PHASE2_EDGE_CLV_MIN
    bcr_ok = (
        bcr is not None
        and bcr_n >= PHASE2_EDGE_BCR_MIN_N
        and float(bcr) >= PHASE2_EDGE_BCR_MIN
    )
    hit_ok = hit_rate is not None and hit_rate >= PHASE2_EDGE_HIT_MIN

    if not hit_ok:
        out["reason"] = f"hit_rate {hit_rate:.1%} < {PHASE2_EDGE_HIT_MIN:.0%}"
        out["suggested_min_edge"] = MIN_EDGE
        return out

    if not (clv_ok or bcr_ok):
        out["reason"] = "CLV/BCR non ancora sufficienti per relax"
        out["suggested_min_edge"] = MIN_EDGE
        return out

    # Graduale: entrambi i segnali forti → floor 2.5%; solo uno → step 2.75%
    strong = hit_ok and clv_ok and bcr_ok
    if strong or (hit_ok and clv_ok and avg_clv is not None and avg_clv >= 0.005):
        suggested = PHASE2_EDGE_FLOOR
        reason = "phase2_full_relax"
    else:
        suggested = PHASE2_EDGE_STEP
        reason = "phase2_partial_relax"

    out["suggested_min_edge"] = round(float(suggested), 4)
    out["unlocked"] = True
    out["reason"] = reason
    return out


def apply_dynamic_edge_to_online_learn(
    ol: dict[str, Any],
    *,
    settled_bets: list[dict[str, Any]] | None = None,
    bcr_betfair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggiorna ``ol`` con min_edge_suggested / phase2 meta se giustificato.

    Non alza sopra CIRCUIT_BREAKER_MIN_EDGE; non abbassa sotto PHASE2_EDGE_FLOOR.
    Se BCR già ha imposto raise, rispetta il raise.
    """
    eval_ = evaluate_dynamic_edge(settled_bets=settled_bets, bcr_betfair=bcr_betfair)
    ol["phase2_edge"] = eval_

    if ol.get("bcr_adjustment") == "raised_min_edge_low_bcr":
        # Stress BCR: non rilassare
        return ol

    if not eval_.get("unlocked"):
        # Mantieni suggested esistente ma non sotto MIN_EDGE se non unlocked
        cur = float(ol.get("min_edge_suggested") or MIN_EDGE)
        ol["min_edge_suggested"] = max(MIN_EDGE, min(cur, CIRCUIT_BREAKER_MIN_EDGE))
        return ol

    suggested = float(eval_["suggested_min_edge"])
    cur = float(ol.get("min_edge_suggested") or MIN_EDGE)
    # Solo allentare (mai alzare qui — raise è gestito da BCR/ROI altrove)
    ol["min_edge_suggested"] = round(
        max(PHASE2_EDGE_FLOOR, min(cur, suggested, CIRCUIT_BREAKER_MIN_EDGE)),
        4,
    )
    if suggested < cur:
        ol["phase2_edge_adjustment"] = eval_.get("reason")
    return ol


def effective_edge_floor() -> float:
    """Floor runtime: PHASE2_EDGE_FLOOR se online_learn ha unlocked, altrimenti MIN_EDGE."""
    try:
        from modules.advisor.online_learn import _load_cal

        ol = (_load_cal().get("online_learn") or {})
        ph = ol.get("phase2_edge") or {}
        if ph.get("unlocked") and float(ol.get("min_edge_suggested") or MIN_EDGE) < MIN_EDGE + 1e-9:
            return float(PHASE2_EDGE_FLOOR)
        if ph.get("unlocked"):
            return float(PHASE2_EDGE_FLOOR)
    except Exception:
        pass
    return float(MIN_EDGE)
