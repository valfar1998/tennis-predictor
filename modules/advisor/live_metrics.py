"""Metriche fase live: BCR Pinnacle, slippage alert, audit paper trading."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "data" / "processed" / "live_metrics.json"
BCR_TARGET = 0.55


def _is_pinnacle_close(source: str | None) -> bool:
    s = str(source or "").lower()
    if not s:
        return False
    if "betfair" in s or "oddssafari" in s or "arbworld" in s:
        return False
    return any(k in s for k in ("pinnacle", "ps_close", "psw", "psl", "tennis-data"))


def compute_bcr(*, pinnacle_only: bool = True) -> dict[str, Any]:
    """Beat Closing Rate su pick settle con chiusura disponibile."""
    from modules.data_update.history import load_history

    rows = load_history(limit=5000)
    settled = [r for r in rows if r.get("hit") is not None and r.get("action") == "bet"]

    pool: list[dict] = []
    for r in settled:
        if r.get("beat_close") is None:
            continue
        src = r.get("close_source")
        if pinnacle_only and not _is_pinnacle_close(src):
            continue
        pool.append(r)

    n = len(pool)
    beats = sum(1 for r in pool if int(r.get("beat_close") or 0) == 1)
    rate = beats / n if n else None

    clv_vals = [float(r["clv"]) for r in pool if r.get("clv") is not None]
    avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else None

    return {
        "n": n,
        "beats": beats,
        "bcr": round(rate, 4) if rate is not None else None,
        "bcr_pct": round(rate * 100, 1) if rate is not None else None,
        "target": BCR_TARGET,
        "target_pct": round(BCR_TARGET * 100, 1),
        "pass": None if n == 0 else rate >= BCR_TARGET,
        "avg_clv": round(avg_clv, 4) if avg_clv is not None else None,
        "pinnacle_only": pinnacle_only,
    }


def compute_execution_summary() -> dict[str, Any]:
    """ROI / hit rate (secondario) + BCR (KPI primario) + slippage Telegram."""
    from modules.data_update.history import history_summary
    from modules.advisor.slippage_audit import slippage_summary

    hist = history_summary()
    settled_n = hist.get("n_settled") or 0

    bcr_pin = compute_bcr(pinnacle_only=True)
    bcr_all = compute_bcr(pinnacle_only=False)

    roi_note = (
        "ROI primi 100 bet guidato dalla varianza - usare BCR Pinnacle come KPI edge"
        if settled_n < 100
        else "Campione >=100: ROI diventa informativo oltre al BCR"
    )

    return {
        "phase": "paper_trading",
        "n_settled": settled_n,
        "n_pending": hist.get("n_pending"),
        "hit_rate": hist.get("hit_rate"),
        "roi_note": roi_note,
        "bcr_pinnacle": bcr_pin,
        "bcr_all_sources": bcr_all,
        "slippage": slippage_summary(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_live_metrics(report: dict[str, Any] | None = None) -> Path:
    report = report or compute_execution_summary()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return METRICS_PATH


def run_live_audit(*, refresh_slippage: bool = True) -> dict[str, Any]:
    """Audit completo fase live (chiamata da main.py metrics / predict)."""
    if refresh_slippage:
        try:
            from modules.advisor.slippage_audit import refresh_slippage_snapshots

            refresh_slippage_snapshots()
        except Exception as exc:
            pass  # non bloccare audit

    report = compute_execution_summary()
    save_live_metrics(report)
    return report


def format_bcr_status(bcr: dict[str, Any]) -> str:
    if not bcr.get("n"):
        return "BCR Pinnacle: nessun pick settle con chiusura Pinnacle ancora"
    pct = bcr.get("bcr_pct")
    flag = "OK" if bcr.get("pass") else "SOTTO TARGET"
    return (
        f"BCR Pinnacle: {pct}% ({bcr['beats']}/{bcr['n']}) "
        f"target >{bcr['target_pct']}% — {flag}"
    )
