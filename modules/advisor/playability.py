"""Indice giocabilità 0–100 (stile football-predictor unified score)."""

from __future__ import annotations

from typing import Any

from modules.constants import KELLY_CAP, MIN_EDGE, MISSING_SIGNAL_SCORE, ODDS_VARIANCE_REF

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


def _odds_sustainability(odds: float) -> float:
    """1.0 su quota ~ref, scende su quote lunghe (varianza alta)."""
    if odds <= 1.01:
        return 0.0
    # 1.50→~1.0, 2.0→1.0, 3.0→0.72, 4.9→0.48, 8→0.35
    ratio = ODDS_VARIANCE_REF / max(odds, 1.01)
    return _clip(0.25 + 0.75 * min(1.0, ratio))


def _value_score(rec: dict | None) -> float:
    """Value/EV penalizzato dalla varianza della quota (non EV grezzo)."""
    if not rec:
        return 0.0
    ev = float(rec.get("ev") or 0)
    edge = float(rec.get("edge_pp") or 0)
    odds = float(rec.get("odds") or 0)
    ev_n = _clip((ev - MIN_EDGE) / 0.12)
    edge_n = _clip(edge / 0.08)
    # Preferisci Kelly-adjusted / Sharpe se presenti
    sharpe = rec.get("odds_sharpe")
    if sharpe is not None and float(sharpe) > 0:
        sharpe_n = _clip(float(sharpe) / 0.20)
    else:
        sharpe_n = ev_n
    sustain = _odds_sustainability(odds)
    raw = 0.35 * ev_n + 0.25 * edge_n + 0.25 * sharpe_n + 0.15 * sustain
    return _clip(raw * (0.55 + 0.45 * sustain))


def _kelly_score(rec: dict | None) -> float:
    if not rec:
        return 0.0
    k = float(rec.get("kelly") or 0)
    if k <= 0:
        return 0.0
    cap = float(rec.get("kelly_cap") or KELLY_CAP)
    base = _clip(k / max(cap, 1e-6))
    adj = rec.get("kelly_adj_rank")
    if adj is not None and float(adj) > 0:
        # Normalizza rank tipico (Kelly capped * sustain ~ 0–cap)
        adj_n = _clip(float(adj) / max(cap, 1e-6))
        return _clip(0.55 * base + 0.45 * adj_n)
    odds = float(rec.get("odds") or 0)
    return _clip(base * _odds_sustainability(odds))


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
        return MISSING_SIGNAL_SCORE, {"missing": True, "source": "arbworld"}
    from modules.advisor.market_timing import apply_temporal_weight, minutes_until_match

    pick = str(rec.get("player") or "")
    pa, pb = str(pred.get("player_a") or ""), str(pred.get("player_b") or "")
    from modules.data_update.entity_resolution import _last_name

    on_a = _last_name(pick) == _last_name(pa) or pick == pa
    vol = moneyway.get("volume_pct_a") if on_a else moneyway.get("volume_pct_b")
    if vol is None:
        return MISSING_SIGNAL_SCORE, {"missing": True, "source": "arbworld"}
    vol_f = float(vol) / 100.0
    score = _clip((vol_f - 0.35) / 0.55)
    total = moneyway.get("total_volume_gbp")
    liq = _clip((float(total or 0) / 5000.0)) if total else 0.4
    raw = _clip(0.7 * score + 0.3 * liq)

    ctx = {**pred, **moneyway}
    mins = minutes_until_match(ctx, match_start=pred.get("_match_start_dt"))
    weighted, tw = apply_temporal_weight(raw, mins)
    return weighted, {
        "volume_pct_pick": vol,
        "total_volume_gbp": total,
        "source": "arbworld",
        "missing": False,
        "temporal_weight": tw,
        "minutes_to_start": mins,
    }


def _dropping_score(
    pred: dict,
    rec: dict | None,
    dropping: dict | None,
) -> tuple[float, dict]:
    if not dropping or not rec:
        return MISSING_SIGNAL_SCORE, {"missing": True, "source": "oddssafari"}
    from modules.advisor.advise import steam_eroded_reasons

    if steam_eroded_reasons(rec, dropping_row=dropping):
        return 0.15, {
            "drop_pct": float(dropping.get("drop_pct") or 0),
            "aligned_with_pick": True,
            "steam_eroded": True,
            "missing": False,
            "source": "oddssafari",
        }

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

    from modules.advisor.market_timing import apply_temporal_weight, minutes_until_match

    ctx = {**pred, **(dropping or {})}
    mins = minutes_until_match(ctx, match_start=pred.get("_match_start_dt"))
    weighted, tw = apply_temporal_weight(score, mins)
    return weighted, {
        "drop_pct": drop,
        "aligned_with_pick": aligned,
        "source": "oddssafari",
        "missing": False,
        "temporal_weight": tw,
        "minutes_to_start": mins,
    }


def compute_playability(
    advised: dict,
    *,
    moneyway_row: dict | None = None,
    dropping_row: dict | None = None,
) -> dict[str, Any]:
    """Calcola giocabilità 0–100 e componenti."""
    rec = advised.get("recommended") or advised.get("best_play")
    pred = advised

    value_n = _value_score(rec)
    agree_n = _model_agreement(advised)
    kelly_n = _kelly_score(rec)
    market_n = _market_quality(advised, rec)
    mw_n, mw_info = _moneyway_score(advised, rec, moneyway_row)
    drop_n, drop_info = _dropping_score(advised, rec, dropping_row)

    # Se MW/Drop mancano, ridistribuisci il peso su value/kelly/market (no padding neutro)
    mw_missing = bool(mw_info.get("missing"))
    drop_missing = bool(drop_info.get("missing"))
    w_value, w_agree, w_kelly, w_market, w_mw, w_drop = 0.28, 0.16, 0.16, 0.15, 0.13, 0.12
    if mw_missing and drop_missing:
        freed = w_mw + w_drop
        w_mw = w_drop = 0.0
        w_value += freed * 0.45
        w_kelly += freed * 0.35
        w_market += freed * 0.20
    elif mw_missing:
        w_value += w_mw * 0.5
        w_kelly += w_mw * 0.5
        w_mw = 0.0
        mw_n = 0.0
    elif drop_missing:
        w_value += w_drop * 0.5
        w_kelly += w_drop * 0.5
        w_drop = 0.0
        drop_n = 0.0

    raw = (
        w_value * value_n
        + w_agree * agree_n
        + w_kelly * kelly_n
        + w_market * market_n
        + w_mw * mw_n
        + w_drop * drop_n
    )
    try:
        from modules.advisor.online_learn import learned_playability_adjustment

        raw += learned_playability_adjustment(advised)
    except Exception:
        pass
    score = round(100 * _clip(raw), 1)

    action = advised.get("action")
    if action == "review":
        score = min(score, 72.0)  # sotto soglia alert Strong
    elif action != "bet":
        score = min(score, 55.0)
    if rec and float(rec.get("ev") or 0) < MIN_EDGE:
        score = min(score, 45.0)
    # Cap soft su quote molto lunghe anche se EV alto
    if rec and float(rec.get("odds") or 0) >= 4.0:
        score = min(score, 78.0)

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
            "moneyway": round(mw_n if not mw_missing else MISSING_SIGNAL_SCORE, 3),
            "dropping_odds": round(drop_n if not drop_missing else MISSING_SIGNAL_SCORE, 3),
            "odds_sustain": round(_odds_sustainability(float((rec or {}).get("odds") or 0)), 3),
            "signals_missing": mw_missing or drop_missing,
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
    rec = advised.get("recommended") or advised.get("best_play")
    pick_side = rec.get("side") if rec else None

    mw = lookup_moneyway(pa, pb, rows=moneyway_rows)
    drop = lookup_dropping(pa, pb, pick_side=pick_side, rows=dropping_rows)
    info = compute_playability(advised, moneyway_row=mw, dropping_row=drop)
    advised.update(info)
    return advised
