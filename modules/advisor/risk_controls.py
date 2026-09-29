"""Controlli rischio esecuzione: Kelly dinamico, circuit breaker, esposizione giornaliera."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.constants import (
    CIRCUIT_BREAKER_KELLY_SCALE,
    CIRCUIT_BREAKER_METRICS_FROM,
    CIRCUIT_BREAKER_MIN_EDGE,
    DAILY_EXPOSURE_CAP,
    DAILY_EXPOSURE_MIN_BETS,
    DRAWDOWN_BREAKER_PCT,
    KELLY_CAP,
    KELLY_CAP_BY_LEVEL,
    MIN_EDGE,
    STREAK_LOSS_UNITS,
    UNIT_SIZE,
)

ROOT = Path(__file__).resolve().parents[2]
RISK_STATE_PATH = ROOT / "data" / "processed" / "risk_state.json"


def infer_tourney_level(tourney: str | None, tourney_level: str | None = None) -> str:
    """Codice livello torneo Sackmann (G/M/A/C/S) da campo o nome evento."""
    if tourney_level:
        code = str(tourney_level).strip().upper()
        if code:
            return code[0]
    t = str(tourney or "").lower()
    if any(k in t for k in ("us open", "wimbledon", "roland garros", "australian open", "grand slam")):
        return "G"
    if any(
        k in t
        for k in (
            "masters",
            "1000",
            "miami",
            "indian wells",
            "monte carlo",
            "madrid",
            "rome",
            "cincinnati",
            "shanghai",
            "paris",
            "canada",
            "montreal",
            "toronto",
        )
    ):
        return "M"
    if "challenger" in t:
        return "C"
    if any(k in t for k in ("itf", "w15", "w25", "w35", "w50", "w75", "w100", "m15", "m25")):
        return "S"
    if "finals" in t and "challenger" not in t:
        return "F"
    return "A"


def kelly_cap_for_prediction(prediction: dict[str, Any]) -> float:
    """Cap Kelly per liquidità / rischio informativo del torneo."""
    level = infer_tourney_level(
        prediction.get("tourney"),
        prediction.get("tourney_level"),
    )
    return float(KELLY_CAP_BY_LEVEL.get(level, KELLY_CAP_BY_LEVEL.get("A", KELLY_CAP)))


def _bet_day(bet: dict[str, Any]) -> str:
    return str(bet.get("date") or bet.get("settled_at") or bet.get("saved_at") or "")[:10]


def _load_settled_bets(*, metrics_from: str | None = None) -> list[dict[str, Any]]:
    """Pick settle action=bet; opzionale filtro data per reset metriche CB (Fase 1)."""
    from modules.data_update.history import load_history

    rows = load_history(limit=2000)
    settled = [r for r in rows if r.get("hit") is not None and r.get("action") == "bet"]
    cutoff = (metrics_from if metrics_from is not None else CIRCUIT_BREAKER_METRICS_FROM) or ""
    cutoff = str(cutoff).strip()[:10]
    if cutoff:
        settled = [r for r in settled if _bet_day(r) >= cutoff]
    settled.sort(key=lambda r: str(r.get("settled_at") or r.get("saved_at") or ""))
    return settled


def _bankroll_metrics(bets: list[dict[str, Any]]) -> dict[str, Any]:
    bankroll = 1.0
    peak = 1.0
    max_dd = 0.0
    streak_loss_units = 0.0
    max_streak_units = 0.0
    consecutive_losses = 0

    for bet in bets:
        stake = float(bet.get("kelly") or 0.0)
        odds = float(bet.get("odds") or 0.0)
        if stake <= 0 or odds <= 1.01:
            continue
        hit = int(bet.get("hit") or 0)
        if hit:
            bankroll += stake * (odds - 1.0)
            streak_loss_units = 0.0
            consecutive_losses = 0
        else:
            bankroll -= stake
            streak_loss_units += stake / UNIT_SIZE
            consecutive_losses += 1
            max_streak_units = max(max_streak_units, streak_loss_units)
        peak = max(peak, bankroll)
        if peak > 0:
            max_dd = max(max_dd, (peak - bankroll) / peak)

    current_dd = (peak - bankroll) / peak if peak > 0 else 0.0
    return {
        "bankroll": round(bankroll, 4),
        "peak": round(peak, 4),
        "max_drawdown": round(max_dd, 4),
        "current_drawdown": round(current_dd, 4),
        "streak_loss_units": round(streak_loss_units, 2),
        "max_streak_loss_units": round(max_streak_units, 2),
        "consecutive_losses": consecutive_losses,
        "n_settled": len(bets),
    }


def circuit_breaker_status(*, min_settled: int = 5) -> dict[str, Any]:
    """Valuta drawdown / streak su finestra post-reset; stress → EV floor soft + Kelly scale."""
    bets = _load_settled_bets()
    metrics = _bankroll_metrics(bets)

    streak_trigger = metrics["streak_loss_units"] >= STREAK_LOSS_UNITS
    dd_trigger = metrics["current_drawdown"] >= DRAWDOWN_BREAKER_PCT
    active = metrics["n_settled"] >= min_settled and (streak_trigger or dd_trigger)

    status = {
        "active": active,
        "min_edge": CIRCUIT_BREAKER_MIN_EDGE if active else MIN_EDGE,
        "base_min_edge": MIN_EDGE,
        "stress_min_edge": CIRCUIT_BREAKER_MIN_EDGE,
        # Fase 1: sotto stress riduci stake, non solo alzare EV
        "kelly_scale": CIRCUIT_BREAKER_KELLY_SCALE if active else 1.0,
        "triggers": {
            "streak_loss_units": streak_trigger,
            "drawdown": dd_trigger,
        },
        "thresholds": {
            "streak_loss_units": STREAK_LOSS_UNITS,
            "drawdown_pct": DRAWDOWN_BREAKER_PCT,
        },
        "metrics_from": CIRCUIT_BREAKER_METRICS_FROM,
        **metrics,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        RISK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RISK_STATE_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    except Exception:
        pass

    return status


def get_risk_context() -> dict[str, Any]:
    """Contesto rischio per predict: online learn + circuit breaker."""
    from modules.advisor.online_learn import effective_min_edge

    cb = circuit_breaker_status()
    learned = effective_min_edge()
    if cb["active"]:
        min_edge = max(learned, CIRCUIT_BREAKER_MIN_EDGE)
    else:
        min_edge = learned
    return {
        "min_edge": min_edge,
        "min_edge_learned": learned,
        "kelly_scale": float(cb.get("kelly_scale") or 1.0),
        "circuit_breaker": cb,
    }


def apply_circuit_breaker_kelly_scale(
    predictions: list[dict[str, Any]],
    *,
    kelly_scale: float | None = None,
) -> list[dict[str, Any]]:
    """Sotto CB attivo: riduce Kelly dei bet (stake), senza cambiare action/storico hit.

    Non demote a no_bet se Kelly scende sotto MIN_KELLY: il filtro edge è già passato;
    lo scale è solo sizing sotto stress.
    """
    scale = float(kelly_scale if kelly_scale is not None else 1.0)
    if scale >= 0.999:
        return predictions

    for pred in predictions:
        if pred.get("action") != "bet":
            continue
        rec = pred.get("recommended")
        if not rec:
            continue
        # Idempotente: già scalato in _archive_advised
        if rec.get("kelly_pre_cb_scale") is not None:
            continue
        old_k = float(rec.get("kelly") or 0.0)
        if old_k <= 0:
            continue
        rec["kelly_pre_cb_scale"] = old_k
        rec["kelly"] = round(old_k * scale, 4)
        if pred.get("best_play") is rec or (
            pred.get("best_play") and pred["best_play"].get("player") == rec.get("player")
        ):
            bp = pred.get("best_play")
            if bp is not None and bp is not rec:
                bp["kelly_pre_cb_scale"] = old_k
                bp["kelly"] = rec["kelly"]
        meta = pred.setdefault("risk_controls", {})
        meta["circuit_breaker_kelly_scaled"] = True
        meta["circuit_breaker_kelly_scale"] = round(scale, 4)

    return predictions


def apply_daily_exposure_limits(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Limita esposizione correlata: stesso giorno + stesso torneo, ≥6 bet."""
    from collections import defaultdict

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, pred in enumerate(predictions):
        if pred.get("action") != "bet":
            continue
        rec = pred.get("recommended")
        if not rec or float(rec.get("kelly") or 0) <= 0:
            continue
        day = str(pred.get("date") or "")[:10]
        tourney = str(pred.get("tourney") or "").strip().lower()
        if not day or not tourney:
            continue
        groups[(day, tourney)].append(idx)

    for (day, tourney), indices in groups.items():
        if len(indices) < DAILY_EXPOSURE_MIN_BETS:
            continue
        total_kelly = sum(float(predictions[i]["recommended"]["kelly"]) for i in indices)
        if total_kelly <= DAILY_EXPOSURE_CAP:
            continue

        scale = DAILY_EXPOSURE_CAP / total_kelly
        for i in indices:
            rec = predictions[i]["recommended"]
            old_k = float(rec["kelly"])
            rec["kelly"] = round(old_k * scale, 4)
            rec["kelly_pre_scale"] = old_k
            meta = predictions[i].setdefault("risk_controls", {})
            meta["daily_exposure_scaled"] = True
            meta["daily_exposure_scale"] = round(scale, 4)
            meta["daily_exposure_group"] = {"date": day, "tourney": tourney, "n_bets": len(indices)}

    return predictions
