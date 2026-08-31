"""Apprendimento da pick chiuse in our_history.sqlite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
CAL_PATH = MODELS / "calibration.json"
REPORT_PATH = MODELS / "online_learn_report.json"
MIN_SETTLED = 12


def _load_cal() -> dict:
    if CAL_PATH.is_file():
        try:
            return json.loads(CAL_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cal(cal: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    CAL_PATH.write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")


def _roi(rows: list[dict]) -> float | None:
    pnl = 0.0
    n = 0
    for r in rows:
        q = r.get("odds")
        if not q or float(q) <= 1.01:
            continue
        n += 1
        pnl += (float(q) - 1.0) if int(r.get("hit") or 0) == 1 else -1.0
    return round(pnl / n, 4) if n >= 5 else None


def learn_from_settled(*, force: bool = False) -> dict[str, Any]:
    from modules.data_update.history import load_history

    rows = load_history(limit=2000)
    settled = [r for r in rows if r.get("hit") is not None and r.get("action") == "bet"]
    report: dict[str, Any] = {
        "ok": False,
        "n_settled": len(settled),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    if len(settled) < MIN_SETTLED and not force:
        report["error"] = f"servono >={MIN_SETTLED} pick chiuse (ora {len(settled)})"
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    hits = sum(int(r.get("hit") or 0) for r in settled)
    report["hit_rate"] = round(hits / len(settled), 4)
    report["roi_all"] = _roi(settled)

    by_band: dict[str, list] = {}
    for r in settled:
        band = str(r.get("playability_band") or "unknown")
        by_band.setdefault(band, []).append(r)

    band_stats = {}
    for band, chunk in by_band.items():
        h = sum(int(x.get("hit") or 0) for x in chunk)
        band_stats[band] = {
            "n": len(chunk),
            "hit_rate": round(h / len(chunk), 4) if chunk else None,
            "roi": _roi(chunk),
        }
    report["by_band"] = band_stats

    mw_aligned = [r for r in settled if r.get("moneyway_vol_pct") is not None]
    drop_aligned = [r for r in settled if int(r.get("dropping_aligned") or 0) == 1]
    drop_against = [
        r for r in settled
        if r.get("dropping_pct") and int(r.get("dropping_aligned") or 0) == 0
    ]
    report["signals"] = {
        "moneyway_n": len(mw_aligned),
        "moneyway_hit_rate": round(
            sum(int(r.get("hit") or 0) for r in mw_aligned) / len(mw_aligned), 4
        ) if mw_aligned else None,
        "dropping_aligned_n": len(drop_aligned),
        "dropping_aligned_hit_rate": round(
            sum(int(r.get("hit") or 0) for r in drop_aligned) / len(drop_aligned), 4
        ) if drop_aligned else None,
        "dropping_against_hit_rate": round(
            sum(int(r.get("hit") or 0) for r in drop_against) / len(drop_against), 4
        ) if drop_against else None,
    }

    cal = _load_cal()
    ol = cal.get("online_learn") or {}

    strong = band_stats.get("strong", {})
    premium = band_stats.get("premium", {})
    if (strong.get("n") or 0) + (premium.get("n") or 0) >= 8:
        sp_hit = (
            (strong.get("hit_rate") or 0) * (strong.get("n") or 0)
            + (premium.get("hit_rate") or 0) * (premium.get("n") or 0)
        ) / max(1, (strong.get("n") or 0) + (premium.get("n") or 0))
        ol["alert_min_suggested"] = 75 if sp_hit >= 0.52 else 80

    sig = report["signals"]
    if sig.get("dropping_aligned_hit_rate") and sig.get("dropping_aligned_n", 0) >= 5:
        ol["dropping_boost"] = min(0.08, max(0.0, sig["dropping_aligned_hit_rate"] - 0.5) * 0.15)
    if sig.get("moneyway_hit_rate") and sig.get("moneyway_n", 0) >= 5:
        ol["moneyway_boost"] = min(0.06, max(0.0, sig["moneyway_hit_rate"] - 0.5) * 0.12)

    if report.get("roi_all") is not None and report["roi_all"] < -0.05:
        ol["min_edge_suggested"] = 0.035
    elif report.get("hit_rate", 0) >= 0.55:
        ol["min_edge_suggested"] = 0.025
    else:
        ol["min_edge_suggested"] = 0.03

    ol["last_n_settled"] = len(settled)
    ol["updated_at"] = report["fitted_at"]
    cal["online_learn"] = ol
    _save_cal(cal)

    report["online_learn"] = ol
    report["ok"] = True
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def learned_playability_boost(pred: dict) -> float:
    """Piccolo boost 0–0.08 da segnali mercato se online_learn lo suggerisce."""
    cal = _load_cal()
    ol = cal.get("online_learn") or {}
    boost = 0.0
    sig = pred.get("market_signals") or {}
    if int(sig.get("aligned_with_pick") or 0) == 1 and ol.get("dropping_boost"):
        boost += float(ol["dropping_boost"])
    if sig.get("volume_pct_pick") is not None and ol.get("moneyway_boost"):
        boost += float(ol["moneyway_boost"]) * 0.5
    return min(0.08, boost)
