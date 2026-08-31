"""Indice giocabilità 0–100 (stile football-predictor unified score)."""

from __future__ import annotations

from typing import Any

from modules.constants import KELLY_CAP, MIN_EDGE

BANDS = (
    (0, 30, "no_bet", "No bet"),
    (30, 60, "lean", "Lean"),
    (60, 75, "playable", "Playable"),
    (75, 90, "strong", "Strong"),
    (90, 101, "premium", "Premium"),
)

MIN_PLAY_ALERT = 75  # soglia alert Telegram (Strong+)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _band(score: float) -> dict[str, str]:
    s = _clip(score, 0, 100)
    for lo, hi, key, label in BANDS:
        if lo <= s < hi:
            return {"key": key, "label": label}
    return {"key": "premium", "label": "Premium"}


def _tourney_level_score(tourney: str | None) -> float:
    t = str(tourney or "").lower()
    if any(k in t for k in ("us open", "wimbledon", "roland garros", "australian open", "grand slam")):
        return 1.0
    if any(k in t for k in ("masters", "atp 1000", "wta 1000", "miami", "indian wells")):
        return 0.85
    if "challenger" in t or "itf" in t:
        return 0.45
    if "us open" in t or "wta" in t or "atp" in t:
        return 0.75
    return 0.55


def _model_agreement(pred: dict) -> float:
    p_blend = float(pred.get("p_win_a") or 0.5)
    parts = []
    for key in ("p_markov", "p_elo", "p_ml"):
        v = pred.get(key)
        if v is not None:
            parts.append(abs(float(v) - p_blend))
    if not parts:
        return 0.5
    avg_div = sum(parts) / len(parts)
    return _clip(1.0 - avg_div / 0.15)


def _value_score(rec: dict | None) -> float:
    if not rec:
        return 0.0
    ev = float(rec.get("ev") or 0)
    edge = float(rec.get("edge_pp") or 0)
    ev_n = _clip((ev - MIN_EDGE) / 0.12)
    edge_n = _clip(edge / 0.08)
    return _clip(0.55 * ev_n + 0.45 * edge_n)


def _kelly_score(rec: dict | None) -> float:
    if not rec:
        return 0.0
    k = float(rec.get("kelly") or 0)
    if k <= 0:
        return 0.0
    return _clip(k / max(KELLY_CAP, 1e-6))


def _market_quality(pred: dict, rec: dict | None) -> float:
    src = str(pred.get("odds_source") or "book").lower()
    src_score = {"betfair": 1.0, "pinnacle": 0.95, "ps": 0.95, "b365": 0.7, "book": 0.55}.get(src, 0.5)
    overround = pred.get("value", {}) or {}
    if isinstance(overround, dict):
        ov = overround.get("overround")
    else:
        ov = None
    if ov is None and pred.get("book_odds"):
        oa, ob = pred["book_odds"].get("a"), pred["book_odds"].get("b")
        if oa and ob:
            ov = 1 / float(oa) + 1 / float(ob)
    tight = 0.5
    if ov is not None:
        tight = _clip(1.08 - (float(ov) - 1.0) / 0.12)
    tour = _tourney_level_score(pred.get("tourney"))
    return _clip(0.45 * src_score + 0.30 * tight + 0.25 * tour)


def _moneyway_score(
    pred: dict,
    rec: dict | None,
    moneyway: dict | None,
) -> tuple[float, dict]:
    if not moneyway or not rec:
        return 0.5, {}
    pick = str(rec.get("player") or "")
    pa, pb = str(pred.get("player_a") or ""), str(pred.get("player_b") or "")
    from modules.data_update.entity_resolution import _last_name

    on_a = _last_name(pick) == _last_name(pa) or pick == pa
    vol = moneyway.get("volume_pct_a") if on_a else moneyway.get("volume_pct_b")
    if vol is None:
        return 0.5, {}
    # steam contrarian: se puntiamo l'underdog e ha poco volume ma EV alto → ok
    # se puntiamo il favorito con >70% volume → segnale sharp
    vol_f = float(vol) / 100.0
    score = _clip((vol_f - 0.35) / 0.55)
    total = moneyway.get("total_volume_gbp")
    liq = _clip((float(total or 0) / 5000.0)) if total else 0.4
    return _clip(0.7 * score + 0.3 * liq), {
        "volume_pct_pick": vol,
        "total_volume_gbp": total,
        "source": "arbworld",
    }


def _dropping_score(
    pred: dict,
    rec: dict | None,
    dropping: dict | None,
) -> tuple[float, dict]:
    if not dropping or not rec:
        return 0.5, {}
    drop = float(dropping.get("drop_pct") or 0)
    side = str(dropping.get("side") or "")
    pick_side = rec.get("side")
    aligned = (pick_side == "A" and side == "1") or (pick_side == "B" and side == "2")
    if aligned and drop >= 10:
        score = _clip(0.55 + drop / 40.0)
    elif aligned and drop >= 5:
        score = _clip(0.45 + drop / 50.0)
    elif drop >= 10 and not aligned:
        score = _clip(0.35 - drop / 80.0)
    else:
        score = 0.45
    return score, {
        "drop_pct": drop,
        "aligned_with_pick": aligned,
        "source": "oddssafari",
    }


def compute_playability(
    advised: dict,
    *,
    moneyway_row: dict | None = None,
    dropping_row: dict | None = None,
) -> dict[str, Any]:
    """Calcola giocabilità 0–100 e componenti."""
    rec = advised.get("recommended")
    pred = advised

    value_n = _value_score(rec)
    agree_n = _model_agreement(advised)
    kelly_n = _kelly_score(rec)
    market_n = _market_quality(advised, rec)
    mw_n, mw_info = _moneyway_score(advised, rec, moneyway_row)
    drop_n, drop_info = _dropping_score(advised, rec, dropping_row)

    raw = (
        0.30 * value_n
        + 0.18 * agree_n
        + 0.12 * kelly_n
        + 0.15 * market_n
        + 0.13 * mw_n
        + 0.12 * drop_n
    )
    try:
        from modules.advisor.online_learn import learned_playability_boost

        raw += learned_playability_boost(advised)
    except Exception:
        pass
    score = round(100 * _clip(raw), 1)

    if advised.get("action") != "bet":
        score = min(score, 55.0)
    if rec and float(rec.get("ev") or 0) < MIN_EDGE:
        score = min(score, 45.0)

    band = _band(score)
    return {
        "playability": score,
        "playability_band": band["key"],
        "playability_label": band["label"],
        "playability_parts": {
            "value": round(value_n, 3),
            "model_agreement": round(agree_n, 3),
            "kelly": round(kelly_n, 3),
            "market_quality": round(market_n, 3),
            "moneyway": round(mw_n, 3),
            "dropping_odds": round(drop_n, 3),
        },
        "market_signals": {**mw_info, **drop_info},
    }


def enrich_playability(
    advised: dict,
    *,
    moneyway_rows: list[dict] | None = None,
    dropping_rows: list[dict] | None = None,
) -> dict:
    from modules.data_update.market_signals import lookup_dropping, lookup_moneyway

    pa = str(advised.get("player_a") or "")
    pb = str(advised.get("player_b") or "")
    rec = advised.get("recommended")
    pick_side = rec.get("side") if rec else None

    mw = lookup_moneyway(pa, pb, rows=moneyway_rows)
    drop = lookup_dropping(pa, pb, pick_side=pick_side, rows=dropping_rows)
    info = compute_playability(advised, moneyway_row=mw, dropping_row=drop)
    advised.update(info)
    return advised
