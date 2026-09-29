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
        "itf": metrics.get("itf_governance"),
        "risk": (metrics.get("governance") or {}),
    }

    # Sintesi giornaliera recente
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
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def format_health_banner(report: dict[str, Any] | None = None) -> str:
    report = report or _load_json(REPORT_PATH) or build_health_report()
    s = report.get("summary") or {}
    return (
        f"HEALTH · bet7d={s.get('bets_last_7d', 0)} shadow7d={s.get('shadow_last_7d', 0)} "
        f"tg14d={s.get('telegram_alerts_last_14d', 0)} "
        f"upcoming_bet={s.get('upcoming_bets', 0)} tg_ready={s.get('upcoming_telegram_ready', 0)} "
        f"BCR_q={s.get('bcr_quality_pct')}% n={s.get('bcr_quality_n')} "
        f"close_missing={s.get('close_missing')} "
        f"phase2_edge={'ON' if s.get('phase2_edge_unlocked') else 'off'}"
    )
