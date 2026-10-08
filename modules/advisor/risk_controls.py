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
    DAILY_BANKROLL_CAP,
    DAILY_EXPOSURE_CAP,
    DAILY_EXPOSURE_MIN_BETS,
    DRAWDOWN_BREAKER_PCT,
    KELLY_CAP,
    KELLY_CAP_BY_LEVEL,
    MIN_EDGE,
    PLAYER_DAY_EXPOSURE_CAP,
    PLAYER_DAY_MIN_BETS,
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
            if rec.get("kelly_pre_scale") is None:
                rec["kelly_pre_scale"] = old_k
            rec["kelly"] = round(old_k * scale, 4)
            meta = predictions[i].setdefault("risk_controls", {})
            meta["daily_exposure_scaled"] = True
            meta["daily_exposure_scale"] = round(scale, 4)
            meta["daily_exposure_group"] = {"date": day, "tourney": tourney, "n_bets": len(indices)}

    return predictions


def _bet_indices(predictions: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    for idx, pred in enumerate(predictions):
        if pred.get("action") != "bet":
            continue
        rec = pred.get("recommended")
        if not rec or float(rec.get("kelly") or 0) <= 0:
            continue
        out.append(idx)
    return out


def _pick_player(pred: dict[str, Any]) -> str:
    rec = pred.get("recommended") or {}
    return str(rec.get("player") or "").strip().lower()


def _fatigue_7d(pred: dict[str, Any]) -> float:
    """Minuti di gioco ~7d sul lato pick (live_features / retirement_context)."""
    feats = pred.get("live_features") or pred.get("features") or {}
    ctx = pred.get("retirement_context") or {}
    pick = _pick_player(pred)
    pa = str(pred.get("player_a") or "").strip().lower()
    if pick and pa and pick == pa:
        v = feats.get("fatigue_minutes_7d_a")
    else:
        v = feats.get("fatigue_minutes_7d_b")
    if v is None:
        v = ctx.get("fatigue_minutes_72h")
        if v is not None:
            return float(v) / 3.0
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def apply_daily_bankroll_cap(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap esposizione totale giornaliera (tutti i tornei) → % bankroll.

    Evita giornate Masters/Slam con troppi match in parallelo che prosciugano il bankroll.
    """
    from collections import defaultdict

    from modules.advisor.staking import scale_to_cap

    by_day: dict[str, list[int]] = defaultdict(list)
    for idx in _bet_indices(predictions):
        day = str(predictions[idx].get("date") or "")[:10]
        if day:
            by_day[day].append(idx)

    for day, indices in by_day.items():
        total = sum(float(predictions[i]["recommended"]["kelly"]) for i in indices)
        scale = scale_to_cap(total, DAILY_BANKROLL_CAP)
        if scale >= 0.999:
            continue
        for i in indices:
            rec = predictions[i]["recommended"]
            old_k = float(rec["kelly"])
            if rec.get("kelly_pre_bankroll_cap") is None:
                rec["kelly_pre_bankroll_cap"] = old_k
            rec["kelly"] = round(old_k * scale, 4)
            meta = predictions[i].setdefault("risk_controls", {})
            meta["daily_bankroll_capped"] = True
            meta["daily_bankroll_scale"] = round(scale, 4)
            meta["daily_bankroll_total_pre"] = round(total, 4)
            meta["daily_bankroll_cap"] = DAILY_BANKROLL_CAP
            meta["daily_bankroll_day"] = day
    return predictions


def apply_player_exposure_limits(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap stake sullo stesso giocatore nello stesso giorno (+ open unsettled in history)."""
    from collections import defaultdict

    from modules.advisor.staking import scale_to_cap

    # Open exposure from history (unsettled bet sullo stesso player/day)
    open_by_key: dict[tuple[str, str], float] = defaultdict(float)
    try:
        from modules.data_update.history import load_history

        for row in load_history(limit=800):
            if row.get("action") != "bet" or row.get("hit") is not None:
                continue
            day = str(row.get("date") or "")[:10]
            pick = str(row.get("pick") or "").strip().lower()
            if day and pick:
                open_by_key[(day, pick)] += float(row.get("kelly") or 0.0)
    except Exception:
        pass

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx in _bet_indices(predictions):
        pred = predictions[idx]
        day = str(pred.get("date") or "")[:10]
        pick = _pick_player(pred)
        if day and pick:
            groups[(day, pick)].append(idx)

    for (day, pick), indices in groups.items():
        if len(indices) < PLAYER_DAY_MIN_BETS and open_by_key.get((day, pick), 0) <= 0:
            # Singola tip senza open history: ok fino al soft corr multiplier
            slate = sum(float(predictions[i]["recommended"]["kelly"]) for i in indices)
            prior = open_by_key.get((day, pick), 0.0)
            if slate + prior <= PLAYER_DAY_EXPOSURE_CAP:
                continue

        slate = sum(float(predictions[i]["recommended"]["kelly"]) for i in indices)
        prior = open_by_key.get((day, pick), 0.0)
        room = max(0.0, PLAYER_DAY_EXPOSURE_CAP - prior)
        scale = scale_to_cap(slate, room if room > 0 else 1e-9)
        if scale >= 0.999:
            continue
        for i in indices:
            rec = predictions[i]["recommended"]
            old_k = float(rec["kelly"])
            if rec.get("kelly_pre_player_cap") is None:
                rec["kelly_pre_player_cap"] = old_k
            rec["kelly"] = round(old_k * scale, 4)
            meta = predictions[i].setdefault("risk_controls", {})
            meta["player_exposure_scaled"] = True
            meta["player_exposure_scale"] = round(scale, 4)
            meta["player_exposure_group"] = {
                "date": day,
                "player": pick,
                "n_bets": len(indices),
                "open_prior": round(prior, 4),
            }
    return predictions


def apply_correlation_kelly_scales(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Moltiplicatori Kelly per correlazione torneo / superficie / stanchezza."""
    from collections import defaultdict

    from modules.advisor.staking import correlation_kelly_multiplier

    # Precompute group stakes
    tourney_stake: dict[tuple[str, str], float] = defaultdict(float)
    surface_n: dict[tuple[str, str], int] = defaultdict(int)
    player_stake: dict[tuple[str, str], float] = defaultdict(float)

    for idx in _bet_indices(predictions):
        pred = predictions[idx]
        day = str(pred.get("date") or "")[:10]
        tourney = str(pred.get("tourney") or "").strip().lower()
        surface = str(pred.get("surface") or "").strip().lower()
        pick = _pick_player(pred)
        k = float(pred["recommended"]["kelly"])
        if day and tourney:
            tourney_stake[(day, tourney)] += k
        if day and surface:
            surface_n[(day, surface)] += 1
        if day and pick:
            player_stake[(day, pick)] += k

    for idx in _bet_indices(predictions):
        pred = predictions[idx]
        rec = pred["recommended"]
        day = str(pred.get("date") or "")[:10]
        tourney = str(pred.get("tourney") or "").strip().lower()
        surface = str(pred.get("surface") or "").strip().lower()
        pick = _pick_player(pred)
        scale, reasons = correlation_kelly_multiplier(
            same_player_day_stake=player_stake.get((day, pick), 0.0),
            same_tourney_day_stake=tourney_stake.get((day, tourney), 0.0),
            same_surface_day_n=surface_n.get((day, surface), 0),
            fatigue_minutes_7d=_fatigue_7d(pred),
        )
        if scale >= 0.999 or not reasons:
            continue
        old_k = float(rec["kelly"])
        if rec.get("kelly_pre_corr_scale") is None:
            rec["kelly_pre_corr_scale"] = old_k
        rec["kelly"] = round(old_k * scale, 4)
        meta = pred.setdefault("risk_controls", {})
        meta["correlation_scaled"] = True
        meta["correlation_scale"] = scale
        meta["correlation_reasons"] = reasons
    return predictions


def apply_portfolio_risk_limits(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pipeline Fase 3: tourney cluster → bankroll day → player → correlazione/fatica.

    Ordine: prima i large-group hard caps, poi soft multipliers.
    Non modifica action / MIN_EDGE.
    """
    predictions = apply_daily_exposure_limits(predictions)
    predictions = apply_daily_bankroll_cap(predictions)
    predictions = apply_player_exposure_limits(predictions)
    predictions = apply_correlation_kelly_scales(predictions)
    return predictions
