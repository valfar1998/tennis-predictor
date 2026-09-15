"""Metriche fase live: BCR Betfair, slippage alert, audit paper trading."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from modules.constants import (
    BCR_ACTIONS,
    BCR_FALLBACK_CLOSE_SOURCES,
    BCR_MIN_CLOSE_DELTA,
)

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "data" / "processed" / "live_metrics.json"
BCR_TARGET = 0.55


def _is_betfair_close(source: str | None) -> bool:
    s = str(source or "").lower()
    return bool(s) and "betfair" in s


def _is_fallback_close(source: str | None) -> bool:
    s = str(source or "").lower()
    if not s:
        return False
    if s in BCR_FALLBACK_CLOSE_SOURCES:
        return True
    return "fallback" in s or "snapshot" in s


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


def _close_odds_for_pick(row: dict[str, Any]) -> float | None:
    from modules.data_update.entity_resolution import _last_name

    ca, cb = row.get("close_odds_a"), row.get("close_odds_b")
    if ca is None or cb is None:
        return None
    pick = str(row.get("pick") or "")
    pa, pb = str(row.get("player_a") or ""), str(row.get("player_b") or "")
    if not pick:
        return None
    side = "A" if _last_name(pick) == _last_name(pa) else "B"
    try:
        return float(ca if side == "A" else cb)
    except (TypeError, ValueError):
        return None


def _is_quality_bcr_row(row: dict[str, Any]) -> bool:
    """Esclude last-LTP ≈ quota bet e fonti fallback (BCR finto ~0%)."""
    src = str(row.get("close_source") or "").lower()
    if not _is_betfair_close(src):
        return False
    if _is_fallback_close(src):
        return False
    odds_bet = row.get("odds")
    close_pick = _close_odds_for_pick(row)
    if odds_bet is None or close_pick is None:
        return True
    try:
        delta = abs(float(close_pick) - float(odds_bet))
    except (TypeError, ValueError):
        return True
    # betfair_settled generico senza delta: spesso era last_ltp riciclato
    if src == "betfair_settled" and delta < BCR_MIN_CLOSE_DELTA:
        return False
    if src in ("betfair_ltp", "betfair_bet_snapshot") and delta < BCR_MIN_CLOSE_DELTA:
        return False
    return True


def compute_bcr(
    *,
    betfair_only: bool = True,
    days: int | None = None,
    actions: tuple[str, ...] | None = None,
    odds_sources: tuple[str, ...] | None = None,
    close_sources: tuple[str, ...] | None = None,
    quality_only: bool = True,
) -> dict[str, Any]:
    """Beat Closing Rate su pick settle con chiusura disponibile."""
    from modules.data_update.history import load_history
    from modules.data_update.entity_resolution import _last_name

    actions = actions or BCR_ACTIONS
    rows = load_history(limit=5000)
    settled = [
        r
        for r in rows
        if r.get("hit") is not None and r.get("action") in actions
    ]
    cutoff = _cutoff_date(days)
    if cutoff is not None:
        settled = [r for r in settled if _row_in_window(r, cutoff)]

    def _src_match(val: str | None, needles: tuple[str, ...] | None) -> bool:
        if not needles:
            return True
        s = str(val or "").lower()
        return any(n.lower() in s for n in needles)

    pool: list[dict] = []
    n_excluded_quality = 0
    for r in settled:
        if r.get("beat_close") is None:
            continue
        src = r.get("close_source")
        if betfair_only and not _is_betfair_close(src):
            continue
        if not _src_match(src, close_sources):
            continue
        if not _src_match(r.get("odds_source"), odds_sources):
            continue
        odds_bet = r.get("odds")
        ca, cb = r.get("close_odds_a"), r.get("close_odds_b")
        pick = str(r.get("pick") or "")
        pa, pb = str(r.get("player_a") or ""), str(r.get("player_b") or "")
        src_l = str(src or "").lower()
        if odds_bet and ca and cb and pick and src_l in (
            "betfair_ltp",
            "betfair_bet_snapshot",
            "betfair_ltp_fallback",
            "kambi_last_odds",
            "kambi_unibet",
        ):
            side = "A" if _last_name(pick) == _last_name(pa) else "B"
            close_pick = float(ca if side == "A" else cb)
            if abs(close_pick - float(odds_bet)) < 0.005:
                continue
        if betfair_only and quality_only and not _is_quality_bcr_row(r):
            n_excluded_quality += 1
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
        "quality_only": bool(quality_only and betfair_only),
        "n_excluded_low_quality": n_excluded_quality if betfair_only and quality_only else 0,
        "actions": list(actions),
    }
    if odds_sources:
        out["odds_sources"] = list(odds_sources)
    if close_sources:
        out["close_sources"] = list(close_sources)
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

    bcr_bf = compute_bcr(
        betfair_only=True, days=bcr_days, actions=BCR_ACTIONS, quality_only=True
    )
    bcr_bf_raw = compute_bcr(
        betfair_only=True, days=bcr_days, actions=BCR_ACTIONS, quality_only=False
    )
    bcr_paper = compute_bcr(
        betfair_only=True,
        days=bcr_days,
        actions=("bet", "shadow", "paper"),
        quality_only=True,
    )
    bcr_kambi = compute_bcr(
        betfair_only=False,
        days=bcr_days,
        actions=("bet", "shadow", "paper"),
        odds_sources=("kambi", "unibet"),
        close_sources=("kambi",),
        quality_only=False,
    )
    bcr_all = compute_bcr(
        betfair_only=False,
        days=bcr_days,
        actions=("bet", "shadow", "paper"),
        quality_only=False,
    )

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
            "BCR Betfair (KPI): BSP / snapshot T−1/T−5/T−60, action=bet|shadow. "
            "Esclusi last-LTP fallback ≈ quota bet. "
            "BCR Kambi (secondario): ingresso Unibet vs snapshot Kambi. "
            "Paper = previsioni valide no_bet."
        ),
        "bcr_betfair": bcr_bf,
        "bcr_betfair_raw": bcr_bf_raw,
        "bcr_kambi": bcr_kambi,
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


def format_bcr_status(bcr: dict[str, Any], *, label: str = "Betfair") -> str:
    window = ""
    if bcr.get("days"):
        window = f" (ultimi {bcr['days']}g: {bcr.get('from_date')} -> {bcr.get('to_date')})"
    if not bcr.get("n"):
        excl = bcr.get("n_excluded_low_quality") or 0
        extra = f" ({excl} escluse low-quality)" if excl else ""
        return f"BCR {label}{window}: nessun pick settle con chiusura {label} nel periodo{extra}"
    pct = bcr.get("bcr_pct")
    flag = "OK" if bcr.get("pass") else "SOTTO TARGET"
    quality = " [quality]" if bcr.get("quality_only") else " [raw]"
    excl = bcr.get("n_excluded_low_quality") or 0
    excl_s = f", esclusi {excl} fallback" if excl else ""
    return (
        f"BCR {label}{window}{quality}: {pct}% ({bcr['beats']}/{bcr['n']}) "
        f"target >{bcr['target_pct']}% — {flag}{excl_s}"
    )
