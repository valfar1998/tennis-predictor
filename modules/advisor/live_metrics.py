"""Metriche fase live: BCR Betfair, slippage alert, audit paper trading."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "data" / "processed" / "live_metrics.json"
BCR_TARGET = 0.55


def _is_betfair_close(source: str | None) -> bool:
    s = str(source or "").lower()
    return bool(s) and "betfair" in s


def _cutoff_date(days: int | None) -> date | None:
    if days is None or days <= 0:
        return None
    return date.today() - timedelta(days=days - 1)


def _row_in_window(row: dict[str, Any], cutoff: date | None) -> bool:
    if cutoff is None:
        return True
    day = str(row.get("date") or "")[:10]
    try:
        return date.fromisoformat(day) >= cutoff
    except ValueError:
        return False


def compute_bcr(
    *,
    betfair_only: bool = True,
    days: int | None = None,
    actions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Beat Closing Rate su pick settle con chiusura disponibile."""
    from modules.data_update.history import load_history
    from modules.data_update.entity_resolution import _last_name

    actions = actions or ("bet",)
    rows = load_history(limit=5000)
    settled = [
        r
        for r in rows
        if r.get("hit") is not None and r.get("action") in actions
    ]
    cutoff = _cutoff_date(days)
    if cutoff is not None:
        settled = [r for r in settled if _row_in_window(r, cutoff)]

    pool: list[dict] = []
    for r in settled:
        if r.get("beat_close") is None:
            continue
        src = r.get("close_source")
        if betfair_only and not _is_betfair_close(src):
            continue
        odds_bet = r.get("odds")
        ca, cb = r.get("close_odds_a"), r.get("close_odds_b")
        pick = str(r.get("pick") or "")
        pa, pb = str(r.get("player_a") or ""), str(r.get("player_b") or "")
        src_l = str(src or "").lower()
        if odds_bet and ca and cb and pick and src_l in ("betfair_ltp", "betfair_bet_snapshot"):
            side = "A" if _last_name(pick) == _last_name(pa) else "B"
            close_pick = float(ca if side == "A" else cb)
            if abs(close_pick - float(odds_bet)) < 0.005:
                continue
        pool.append(r)

    n = len(pool)
    beats = sum(1 for r in pool if int(r.get("beat_close") or 0) == 1)
    rate = beats / n if n else None

    clv_vals = [float(r["clv"]) for r in pool if r.get("clv") is not None]
    avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else None

    out: dict[str, Any] = {
        "n": n,
        "beats": beats,
        "bcr": round(rate, 4) if rate is not None else None,
        "bcr_pct": round(rate * 100, 1) if rate is not None else None,
        "target": BCR_TARGET,
        "target_pct": round(BCR_TARGET * 100, 1),
        "pass": None if n == 0 else rate >= BCR_TARGET,
        "avg_clv": round(avg_clv, 4) if avg_clv is not None else None,
        "betfair_only": betfair_only,
        "actions": list(actions),
    }
    if days is not None:
        out["days"] = int(days)
        out["from_date"] = cutoff.isoformat() if cutoff else None
        out["to_date"] = date.today().isoformat()
    return out


def compute_execution_summary(*, bcr_days: int | None = None) -> dict[str, Any]:
    """ROI / hit rate (secondario) + BCR (KPI primario) + slippage Telegram."""
    from modules.data_update.history import history_summary
    from modules.advisor.slippage_audit import slippage_summary

    hist = history_summary()
    settled_n = hist.get("n_settled") or 0

    bcr_bf = compute_bcr(betfair_only=True, days=bcr_days, actions=("bet",))
    bcr_paper = compute_bcr(betfair_only=True, days=bcr_days, actions=("bet", "paper"))
    bcr_all = compute_bcr(betfair_only=False, days=bcr_days, actions=("bet", "paper"))

    roi_note = (
        "ROI primi 100 bet guidato dalla varianza - usare BCR Betfair come KPI edge"
        if settled_n < 100
        else "Campione >=100: ROI diventa informativo oltre al BCR"
    )

    out = {
        "phase": "paper_trading",
        "n_settled": settled_n,
        "n_pending": hist.get("n_pending"),
        "hit_rate": hist.get("hit_rate"),
        "roi_note": roi_note,
        "bcr_source": "betfair",
        "bcr_note": (
            "Il 3.8% (3/80) citato in chat il 2026-09-02 non era nel DB "
            "(audit già 0/80) e i 3 match non erano nominati: non ricostruibile. "
            "BCR usa solo chiusure distinte dalla quota d'ingresso; paper = previsioni valide no_bet."
        ),
        "bcr_betfair": bcr_bf,
        "bcr_paper": bcr_paper,
        "bcr_all_sources": bcr_all,
        "slippage": slippage_summary(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from modules.advisor.validation_freeze import governance_status
        from modules.advisor.itf_governance import effective_itf_params

        out["governance"] = governance_status()
        out["itf_governance"] = effective_itf_params(refresh=True)
    except Exception:
        pass
    return out


def save_live_metrics(report: dict[str, Any] | None = None, *, bcr_days: int | None = None) -> Path:
    report = report or compute_execution_summary(bcr_days=bcr_days)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return METRICS_PATH


def run_live_audit(*, refresh_slippage: bool = True, bcr_days: int | None = None) -> dict[str, Any]:
    """Audit completo fase live (chiamata da main.py metrics / predict)."""
    try:
        from modules.advisor.validation_freeze import maybe_auto_unfreeze

        maybe_auto_unfreeze()
    except Exception:
        pass

    if refresh_slippage:
        try:
            from modules.advisor.slippage_audit import refresh_slippage_snapshots

            refresh_slippage_snapshots()
        except Exception:
            pass  # non bloccare audit

    report = compute_execution_summary(bcr_days=bcr_days)
    save_live_metrics(report, bcr_days=bcr_days)
    return report


def format_bcr_status(bcr: dict[str, Any]) -> str:
    window = ""
    if bcr.get("days"):
        window = f" (ultimi {bcr['days']}g: {bcr.get('from_date')} -> {bcr.get('to_date')})"
    if not bcr.get("n"):
        return f"BCR Betfair{window}: nessun pick settle con chiusura Betfair nel periodo"
    pct = bcr.get("bcr_pct")
    flag = "OK" if bcr.get("pass") else "SOTTO TARGET"
    return (
        f"BCR Betfair{window}: {pct}% ({bcr['beats']}/{bcr['n']}) "
        f"target >{bcr['target_pct']}% — {flag}"
    )
