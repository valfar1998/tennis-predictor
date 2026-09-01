"""Contatore giornaliero chiamate RapidAPI (quota tier gratuito)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
USAGE_FILE = ROOT / "data" / "raw" / "rapidapi_usage.json"
HISTORY_DAYS = 14


def daily_limit() -> int:
    raw = (os.environ.get("RAPIDAPI_DAILY_LIMIT") or "50").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 50


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _default_state() -> dict[str, Any]:
    return {
        "date": _today(),
        "count": 0,
        "daily_limit": daily_limit(),
        "by_host": {},
        "calls": [],
    }


def _load_state() -> dict[str, Any]:
    if not USAGE_FILE.is_file():
        return _default_state()
    try:
        state = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    if state.get("date") != _today():
        history = list(state.get("history") or [])
        history.append(
            {
                "date": state.get("date"),
                "count": int(state.get("count") or 0),
                "daily_limit": int(state.get("daily_limit") or daily_limit()),
                "by_host": dict(state.get("by_host") or {}),
            }
        )
        state = _default_state()
        state["history"] = history[-HISTORY_DAYS:]
    state["daily_limit"] = daily_limit()
    state.setdefault("by_host", {})
    state.setdefault("calls", [])
    return state


def _save_state(state: dict[str, Any]) -> None:
    calls = list(state.get("calls") or [])
    state["calls"] = calls[-200:]
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def record_call(*, host: str, url: str, status_code: int | None = None) -> dict[str, Any]:
    state = _load_state()
    state["count"] = int(state.get("count") or 0) + 1
    by_host = dict(state.get("by_host") or {})
    by_host[host] = int(by_host.get(host) or 0) + 1
    state["by_host"] = by_host
    state["last_call_at"] = datetime.now(timezone.utc).isoformat()
    state["calls"].append(
        {
            "at": state["last_call_at"],
            "host": host,
            "url": url,
            "status_code": status_code,
        }
    )
    _save_state(state)
    return summarize(state)


def summarize(state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = state or _load_state()
    count = int(state.get("count") or 0)
    limit = int(state.get("daily_limit") or daily_limit())
    remaining = max(0, limit - count)
    pct = round(100.0 * count / limit, 1) if limit else 0.0
    if count >= limit:
        status = "exhausted"
    elif count >= int(limit * 0.8):
        status = "warning"
    else:
        status = "ok"
    return {
        "date": state.get("date") or _today(),
        "count": count,
        "daily_limit": limit,
        "remaining": remaining,
        "pct_used": pct,
        "by_host": dict(state.get("by_host") or {}),
        "last_call_at": state.get("last_call_at"),
        "status": status,
        "history": list(state.get("history") or [])[-HISTORY_DAYS:],
    }


def get_usage_summary() -> dict[str, Any]:
    return summarize()


def estimate_fetch_calls(*, days: int = 5, include_tennis: bool = True, include_sofa: bool = True) -> int:
    """Stima chiamate per un refresh completo (senza cache)."""
    per_day = 0
    if include_tennis:
        per_day += 2
    if include_sofa:
        per_day += 1
    return max(0, (days + 1) * per_day)


def format_usage_line(summary: dict[str, Any] | None = None) -> str:
    s = summary or get_usage_summary()
    return (
        f"RapidAPI oggi: {s['count']}/{s['daily_limit']} "
        f"({s['remaining']} rimanenti, {s['pct_used']}% usato)"
    )
