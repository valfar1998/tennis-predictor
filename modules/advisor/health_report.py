"""Reportistica salute modello — volume bet, Telegram, BCR/close quality (Fase 2)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "data" / "processed" / "health_report.json"
SENT_PATH = ROOT / "data" / "processed" / "telegram_alerts_sent.json"
UPCOMING_PATH = ROOT / "data" / "processed" / "upcoming_predictions.json"
METRICS_PATH = ROOT / "data" / "processed" / "live_metrics.json"


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _daily_bet_volume(*, days: int = 14) -> list[dict[str, Any]]:
    from modules.data_update.history import load_history

    rows = load_history(limit=5000)
    cutoff = date.today() - timedelta(days=days - 1)
    by_day: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        day = str(r.get("date") or "")[:10]
        try:
            if date.fromisoformat(day) < cutoff:
                continue
        except ValueError:
            continue
        action = str(r.get("action") or "no_bet")
        if action in ("bet", "shadow", "paper"):
            by_day[day][action] += 1
            by_day[day]["total"] += 1

    out = []
    for i in range(days):
        d = (date.today() - timedelta(days=days - 1 - i)).isoformat()
        c = by_day.get(d) or Counter()
        out.append(
            {
                "date": d,
                "bet": int(c.get("bet") or 0),
                "shadow": int(c.get("shadow") or 0),
                "paper": int(c.get("paper") or 0),
                "total": int(c.get("total") or 0),
            }
        )
    return out


def _telegram_alert_stats(*, days: int = 14) -> dict[str, Any]:
    sent = _load_json(SENT_PATH) or {}
    if not isinstance(sent, dict):
        return {"n_keys": 0, "n_recent": 0, "by_day": []}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    by_day: Counter = Counter()
    recent = 0
    for _key, ts in sent.items():
        try:
            when = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= cutoff:
            recent += 1
            by_day[when.date().isoformat()] += 1
    return {
        "n_keys": len(sent),
        "n_recent": recent,
        "by_day": [{"date": d, "n": by_day[d]} for d in sorted(by_day)],
    }


def _upcoming_snapshot() -> dict[str, Any]:
    raw = _load_json(UPCOMING_PATH)
    preds = raw if isinstance(raw, list) else (raw or {}).get("predictions") or []
    actions = Counter(str(p.get("action") or "no_bet") for p in preds)
    alertable = 0
    try:
        from modules.notify.alerts import _is_telegram_bet

        alertable = sum(1 for p in preds if _is_telegram_bet(p))
    except Exception:
        alertable = int(actions.get("bet") or 0)
    return {
        "n_matches": len(preds),
        "n_bet": int(actions.get("bet") or 0),
        "n_shadow": int(actions.get("shadow") or 0),
        "n_review": int(actions.get("review") or 0),
        "n_no_bet": int(actions.get("no_bet") or 0),
        "n_telegram_alertable": alertable,
    }


def _equity_curves(*, limit: int = 500) -> dict[str, Any]:
    """Equity unitizzata bet vs shadow (solo settle con odds/kelly)."""
    from modules.data_update.history import load_history

    curves: dict[str, list[dict[str, Any]]] = {"bet": [], "shadow": []}
    bank: dict[str, float] = {"bet": 1.0, "shadow": 1.0}

    rows = load_history(limit=limit)
    rows = sorted(rows, key=lambda r: str(r.get("settled_at") or r.get("saved_at") or r.get("date") or ""))
    for r in rows:
        action = r.get("action")
        if action not in curves or r.get("hit") is None:
            continue
        stake = float(r.get("kelly") or 0.0)
        # Shadow hanno kelly=0: usa EV proxy 1% unit per equity paper
        if action == "shadow" and stake <= 0:
            stake = 0.01
        odds = float(r.get("odds") or 0.0)
        if stake <= 0 or odds <= 1.01:
            continue
        hit = int(r.get("hit") or 0)
        if hit:
            bank[action] += stake * (odds - 1.0)
        else:
            bank[action] -= stake
        curves[action].append(
            {
                "t": str(r.get("settled_at") or r.get("date") or "")[:19],
                "equity": round(bank[action], 4),
                "hit": hit,
            }
        )
    return {
        "bet": curves["bet"],
        "shadow": curves["shadow"],
        "final_bet": bank["bet"],
        "final_shadow": bank["shadow"],
    }


def compute_recommended_roi(*, limit: int = 5000) -> dict[str, Any]:
    """ROI come se avessi seguito i messaggi Telegram **allineati al nuovo metodo**.

    Esclude tip underdog con edge grezzo ≤ noise floor (false-edge pre-calibrazione).
    Per ogni tip settle usa solo ``odds_alert`` / ``kelly_alert`` congelati.
    ROI = Σ P&L / Σ stake (Kelly-pesato).
    """
    from modules.advisor.roi_policy import ROI_POLICY_LABEL, ROI_POLICY_SINCE, filter_roi_tips
    from modules.data_update.history import backfill_odds_alert_from_telegram, load_history

    try:
        backfill = backfill_odds_alert_from_telegram(limit=limit)
    except Exception as exc:
        backfill = {"error": str(exc)}

    rows = load_history(limit=limit)
    bets, policy_stats = filter_roi_tips(rows)
    pending = [r for r in bets if r.get("hit") is None]
    settled = sorted(
        [r for r in bets if r.get("hit") is not None],
        key=lambda r: str(r.get("settled_at") or r.get("saved_at") or r.get("date") or ""),
    )

    def _alert_odds(r: dict) -> float:
        for key in ("odds_alert", "odds"):
            try:
                v = float(r.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if v > 1.01:
                return v
        return 0.0

    def _alert_kelly(r: dict) -> float:
        for key in ("kelly_alert", "kelly"):
            try:
                v = float(r.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
        return 0.0

    def _tip_pnl(odds: float, kelly: float, hit: int) -> float:
        return kelly * (odds - 1.0) if hit else -kelly

    staked = 0.0
    pnl = 0.0
    hits = 0
    n = 0
    n_frozen = 0
    hit_stake = 0.0
    bankroll = 1.0
    curve: list[dict[str, Any]] = []

    for r in settled:
        odds = _alert_odds(r)
        kelly = _alert_kelly(r)
        if odds <= 1.01 or kelly <= 0:
            continue
        if r.get("odds_alert") is not None and float(r.get("odds_alert") or 0) > 1.01:
            n_frozen += 1
        hit = int(r.get("hit") or 0)
        tip_pnl = _tip_pnl(odds, kelly, hit)
        n += 1
        hits += hit
        staked += kelly
        pnl += tip_pnl
        if hit:
            hit_stake += kelly
        bankroll += tip_pnl
        roi_so_far = pnl / staked if staked > 0 else 0.0
        curve.append(
            {
                "t": str(r.get("settled_at") or r.get("date") or "")[:19],
                "n": n,
                "roi": round(roi_so_far, 4),
                "roi_kelly": round(roi_so_far, 4),
                "pnl": round(pnl, 4),
                "staked": round(staked, 4),
                "bankroll": round(bankroll, 4),
                "hit": hit,
                "pick": str(r.get("pick") or ""),
                "odds": round(odds, 3),
                "kelly": round(kelly, 5),
                "odds_frozen": bool(
                    r.get("odds_alert") is not None and float(r.get("odds_alert") or 0) > 1.01
                ),
            }
        )

    def _window(days: int) -> dict[str, Any]:
        cutoff = date.today() - timedelta(days=days - 1)
        w_pnl = w_stake = 0.0
        w_n = w_hits = 0
        for r in settled:
            day = str(r.get("date") or r.get("settled_at") or "")[:10]
            try:
                if date.fromisoformat(day) < cutoff:
                    continue
            except ValueError:
                continue
            odds = _alert_odds(r)
            kelly = _alert_kelly(r)
            if odds <= 1.01 or kelly <= 0:
                continue
            hit = int(r.get("hit") or 0)
            w_n += 1
            w_hits += hit
            w_stake += kelly
            w_pnl += _tip_pnl(odds, kelly, hit)
        return {
            "n": w_n,
            "hits": w_hits,
            "hit_rate": round(w_hits / w_n, 4) if w_n else None,
            "roi": round(w_pnl / w_stake, 4) if w_stake > 0 else None,
            "roi_kelly": round(w_pnl / w_stake, 4) if w_stake > 0 else None,
            "roi_flat": round(w_pnl / w_stake, 4) if w_stake > 0 else None,  # alias UI legacy
            "staked": round(w_stake, 4),
            "pnl": round(w_pnl, 4),
        }

    roi = round(pnl / staked, 4) if staked > 0 else None
    return {
        "ok": True,
        "n_bets": len(bets),
        "n_bets_all": policy_stats.get("n_bets_all"),
        "n_excluded": policy_stats.get("n_excluded"),
        "n_settled": n,
        "n_pending": len(pending),
        "n_odds_frozen": n_frozen,
        "hits": hits,
        "hit_rate": round(hits / n, 4) if n else None,
        "hit_rate_weighted": round(hit_stake / staked, 4) if staked > 0 else None,
        "roi": roi,
        "roi_kelly": roi,
        "roi_flat": roi,  # alias: il ROI ufficiale è Kelly-weighted
        "policy": ROI_POLICY_LABEL,
        "policy_since": ROI_POLICY_SINCE,
        "pnl": round(pnl, 4) if n else None,
        "pnl_units": round(pnl, 4) if n else None,
        "kelly_staked": round(staked, 4),
        "kelly_pnl": round(pnl, 4),
        "bankroll": round(bankroll, 4) if n else None,
        "avg_kelly": round(staked / n, 5) if n else None,
        "last_7d": _window(7),
        "last_30d": _window(30),
        "curve": curve,
        "backfill": backfill,
        "odds_source": "odds_alert + kelly_alert (Telegram freeze)",
        "method": "kelly_weighted",
    }


def build_health_report(*, days: int = 14, refresh_metrics: bool = False) -> dict[str, Any]:
    """Compila report salute modello e lo persiste su disk."""
    if refresh_metrics:
        try:
            from modules.advisor.live_metrics import run_live_audit

            run_live_audit(refresh_slippage=False)
        except Exception:
            pass

    metrics = _load_json(METRICS_PATH) or {}
    bf = metrics.get("bcr_betfair") or {}
    bf_raw = metrics.get("bcr_betfair_raw") or {}
    close_pipe = metrics.get("close_pipeline") or {}
    if not close_pipe:
        try:
            from modules.advisor.live_metrics import close_pipeline_health

            close_pipe = close_pipeline_health()
        except Exception:
            close_pipe = {}
    phase2 = metrics.get("phase2_edge") or {}
    try:
        from modules.advisor.dynamic_edge import evaluate_dynamic_edge
        from modules.advisor.online_learn import effective_min_edge

        if not phase2:
            phase2 = evaluate_dynamic_edge(bcr_betfair=bf)
        min_edge = effective_min_edge()
    except Exception:
        min_edge = None

    try:
        from modules.advisor.risk_controls import circuit_breaker_status

        cb = circuit_breaker_status()
    except Exception:
        cb = {}

    equity = _equity_curves()
    roi = compute_recommended_roi()

    report: dict[str, Any] = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "daily_volume": _daily_bet_volume(days=days),
        "telegram": _telegram_alert_stats(days=days),
        "upcoming": _upcoming_snapshot(),
        "bcr": {
            "betfair_quality": {
                "n": bf.get("n"),
                "bcr_pct": bf.get("bcr_pct"),
                "pass": bf.get("pass"),
                "n_excluded_low_quality": bf.get("n_excluded_low_quality"),
            },
            "betfair_raw": {
                "n": bf_raw.get("n"),
                "bcr_pct": bf_raw.get("bcr_pct"),
            },
            "close_pipeline": close_pipe,
        },
        "edge": {
            "effective_min_edge": min_edge,
            "phase2": phase2,
        },
        "circuit_breaker": {
            "active": bool(cb.get("active")),
            "current_drawdown": cb.get("current_drawdown"),
            "streak_loss_units": cb.get("streak_loss_units"),
            "kelly_scale": cb.get("kelly_scale"),
        },
        "equity": equity,
        "roi": {
            **{k: v for k, v in roi.items() if k != "curve"},
            "curve": roi.get("curve") or [],
        },
        "itf": metrics.get("itf_governance"),
        "risk": (metrics.get("governance") or {}),
    }

    vol = report["daily_volume"]
    last7 = vol[-7:] if len(vol) >= 7 else vol
    report["summary"] = {
        "bets_last_7d": sum(d["bet"] for d in last7),
        "shadow_last_7d": sum(d["shadow"] for d in last7),
        "telegram_alerts_last_14d": report["telegram"]["n_recent"],
        "upcoming_bets": report["upcoming"]["n_bet"],
        "upcoming_telegram_ready": report["upcoming"]["n_telegram_alertable"],
        "bcr_quality_n": bf.get("n"),
        "bcr_quality_pct": bf.get("bcr_pct"),
        "close_missing": close_pipe.get("n_missing_close"),
        "phase2_edge_unlocked": bool((phase2 or {}).get("unlocked")),
        "circuit_breaker_active": bool(cb.get("active")),
        "equity_bet": equity.get("final_bet"),
        "equity_shadow": equity.get("final_shadow"),
        "roi_flat": roi.get("roi"),
        "roi_kelly": roi.get("roi"),
        "roi_n_settled": roi.get("n_settled"),
        "roi_hit_rate": roi.get("hit_rate"),
        "roi_hit_rate_weighted": roi.get("hit_rate_weighted"),
        "roi_pending": roi.get("n_pending"),
        "roi_staked": roi.get("kelly_staked"),
        "roi_pnl": roi.get("kelly_pnl"),
        "roi_bankroll": roi.get("bankroll"),
        "roi_flat_7d": (roi.get("last_7d") or {}).get("roi"),
        "roi_flat_30d": (roi.get("last_30d") or {}).get("roi"),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def format_health_banner(report: dict[str, Any] | None = None) -> str:
    report = report or _load_json(REPORT_PATH) or build_health_report()
    s = report.get("summary") or {}
    roi = s.get("roi_kelly")
    if roi is None:
        roi = s.get("roi_flat")
    roi_txt = f"{100.0 * float(roi):+.1f}%" if roi is not None else "n/d"
    return (
        f"HEALTH · bet7d={s.get('bets_last_7d', 0)} shadow7d={s.get('shadow_last_7d', 0)} "
        f"tg14d={s.get('telegram_alerts_last_14d', 0)} "
        f"upcoming_bet={s.get('upcoming_bets', 0)} tg_ready={s.get('upcoming_telegram_ready', 0)} "
        f"ROI_kelly={roi_txt} n={s.get('roi_n_settled', 0)} "
        f"stake={s.get('roi_staked')} "
        f"BCR_q={s.get('bcr_quality_pct')}% n={s.get('bcr_quality_n')} "
        f"close_missing={s.get('close_missing')} "
        f"phase2_edge={'ON' if s.get('phase2_edge_unlocked') else 'off'} "
        f"CB={'ON' if s.get('circuit_breaker_active') else 'off'}"
    )
