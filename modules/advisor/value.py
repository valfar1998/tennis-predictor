"""Valore scommessa: de-vig Shin/Power, EV, edge vs mercato sharp."""

from __future__ import annotations

from typing import Any

REAL_SOURCES = {"pinnacle", "ps", "betfair", "book", "b365"}


def is_real_odds(source: str | None, odds: float | None) -> bool:
    if odds is None or float(odds) <= 1.01:
        return False
    if not source:
        return False
    return str(source).strip().lower() in REAL_SOURCES


def devig_multiplicative(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Rimozione overround moltiplicativa (2-way)."""
    imp_a, imp_b = 1 / odds_a, 1 / odds_b
    total = imp_a + imp_b
    if total <= 1.0:
        return imp_a, imp_b
    return imp_a / total, imp_b / total


def devig_shin(odds_a: float, odds_b: float, *, max_iter: int = 100) -> tuple[float, float]:
    """Metodo Shin per estrarre probabilità fair da quote con insider trading model."""
    imp_a, imp_b = 1 / odds_a, 1 / odds_b
    z = 0.0
    for _ in range(max_iter):
        denom_a = 1 - z * (1 - imp_a)
        denom_b = 1 - z * (1 - imp_b)
        if denom_a <= 0 or denom_b <= 0:
            break
        p_a = imp_a / denom_a
        p_b = imp_b / denom_b
        total = p_a + p_b
        p_a, p_b = p_a / total, p_b / total
        z_new = sum(imp_i * (1 - p_i) for imp_i, p_i in [(imp_a, p_a), (imp_b, p_b)]) / 2
        if abs(z_new - z) < 1e-8:
            break
        z = z_new
    denom_a = max(1e-9, 1 - z * (1 - imp_a))
    denom_b = max(1e-9, 1 - z * (1 - imp_b))
    p_a = imp_a / denom_a
    p_b = imp_b / denom_b
    total = p_a + p_b
    return p_a / total, p_b / total


def devig_power(odds_a: float, odds_b: float, *, k: float | None = None) -> tuple[float, float]:
    """Power method: trova k tale che sum((1/odds)^k) = 1."""
    imp_a, imp_b = 1 / odds_a, 1 / odds_b
    if k is None:
        lo, hi = 0.5, 2.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if imp_a**mid + imp_b**mid > 1:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2
    p_a = imp_a**k
    p_b = imp_b**k
    total = p_a + p_b
    return p_a / total, p_b / total


def compute_ev(prob: float, odds: float) -> float:
    return prob * odds - 1.0


def enrich_value(
    prediction: dict,
    odds_a: float | None,
    odds_b: float | None,
    *,
    source: str = "book",
    method: str = "shin",
) -> dict:
    """Calcola EV per entrambi i lati del match."""
    p_a = prediction.get("p_win_a", 0.5)
    p_b = 1.0 - p_a
    out: dict[str, Any] = dict(prediction)

    if not odds_a or not odds_b or odds_a <= 1.01 or odds_b <= 1.01:
        out["value"] = None
        return out

    devig_fn = {"shin": devig_shin, "power": devig_power, "multiplicative": devig_multiplicative}
    fn = devig_fn.get(method, devig_shin)
    mkt_a, mkt_b = fn(odds_a, odds_b)

    ev_a = compute_ev(p_a, odds_a)
    ev_b = compute_ev(p_b, odds_b)
    edge_a = p_a - mkt_a
    edge_b = p_b - mkt_b

    picks = []
    for side, p, odds, ev, edge, mkt in [
        ("A", p_a, odds_a, ev_a, edge_a, mkt_a),
        ("B", p_b, odds_b, ev_b, edge_b, mkt_b),
    ]:
        picks.append({
            "side": side,
            "player": prediction.get(f"player_{side.lower()}"),
            "probability": round(p, 4),
            "odds": odds,
            "ev": round(ev, 4),
            "edge_pp": round(edge, 4),
            "mkt_prob": round(mkt, 4),
            "odds_real": is_real_odds(source, odds),
            "odds_source": source,
        })

    best = max(picks, key=lambda x: x["ev"])
    out["value"] = {
        "picks": picks,
        "best": best,
        "devig_method": method,
        "overround": round(1 / odds_a + 1 / odds_b, 4),
    }
    return out
