"""Audit slippage post-alert Telegram (steam entro 3 minuti)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "processed" / "our_history.sqlite"

SLIPPAGE_WINDOW_MIN = 3
STEAM_SLIPPAGE_PCT = 3.0  # quota pick cala >3% entro 3 min

_CREATE_ALERTS = """
CREATE TABLE IF NOT EXISTS alert_log (
    alert_key TEXT PRIMARY KEY,
    sent_at TEXT NOT NULL,
    player_a TEXT,
    player_b TEXT,
    pick TEXT,
    pick_side TEXT,
    odds_at_alert REAL,
    ev_at_alert REAL,
    betfair_event_id TEXT,
    betfair_market_id TEXT,
    odds_t3 REAL,
    t3_recorded_at TEXT,
    slippage_pct REAL,
    steam_within_3m INTEGER,
    match_date TEXT
);
"""


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.executescript(_CREATE_ALERTS)
    cols = {row[1] for row in c.execute("PRAGMA table_info(alert_log)")}
    if "betfair_market_id" not in cols:
        c.execute("ALTER TABLE alert_log ADD COLUMN betfair_market_id TEXT")
        c.commit()
    c.row_factory = sqlite3.Row
    return c


def log_alert(pred: dict, *, sent_at: str | None = None) -> None:
    """Registra odds/EV al momento invio Telegram."""
    rec = pred.get("recommended") or {}
    side = str(rec.get("side") or "")
    key = "|".join(
        str(pred.get(k) or "")
        for k in ("player_a", "player_b", "date", "tourney", "betfair_event_id")
    )
    if not key.strip("|"):
        return

    ts = sent_at or datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO alert_log
            (alert_key, sent_at, player_a, player_b, pick, pick_side,
             odds_at_alert, ev_at_alert, betfair_event_id, betfair_market_id, match_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key,
                ts,
                pred.get("player_a"),
                pred.get("player_b"),
                rec.get("player"),
                side,
                rec.get("odds"),
                rec.get("ev"),
                pred.get("betfair_event_id"),
                pred.get("betfair_market_id"),
                str(pred.get("date") or "")[:10],
            ),
        )
        c.commit()


def _pick_odds_from_event(ev: dict, pick_side: str) -> float | None:
    if pick_side == "A":
        v = ev.get("odd_a")
    elif pick_side == "B":
        v = ev.get("odd_b")
    else:
        return None
    try:
        return float(v) if v and float(v) > 1.01 else None
    except (TypeError, ValueError):
        return None


def refresh_slippage_snapshots(*, window_min: int = SLIPPAGE_WINDOW_MIN) -> dict[str, Any]:
    """Cattura quota T+3min per alert pendenti (best-effort su cache Betfair)."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=window_min)
    updated = 0
    skipped = 0

    try:
        from modules.data_update.betfair import load_betfair_cache, lookup_betfair_match
    except Exception:
        return {"updated": 0, "error": "betfair unavailable"}

    events = load_betfair_cache()

    with _conn() as c:
        rows = c.execute(
            """SELECT * FROM alert_log
               WHERE odds_t3 IS NULL AND datetime(sent_at) <= datetime(?)""",
            (cutoff.isoformat(),),
        ).fetchall()

        for row in rows:
            rec = dict(row)
            sent = datetime.fromisoformat(str(rec["sent_at"]).replace("Z", "+00:00"))
            if sent.tzinfo is None:
                sent = sent.replace(tzinfo=timezone.utc)
            if now < sent + timedelta(minutes=window_min):
                continue

            side = str(rec.get("pick_side") or "")
            odds_alert = float(rec.get("odds_at_alert") or 0)
            if odds_alert <= 1.01:
                skipped += 1
                continue

            bf = lookup_betfair_match(
                str(rec.get("player_a") or ""),
                str(rec.get("player_b") or ""),
                events=events,
                match_date=str(rec.get("match_date") or "")[:10],
            )
            if not bf:
                skipped += 1
                continue

            odds_t3 = _pick_odds_from_event(bf, side)
            if not odds_t3:
                skipped += 1
                continue

            slip = (odds_alert - odds_t3) / odds_alert * 100.0
            steam = 1 if slip >= STEAM_SLIPPAGE_PCT else 0

            c.execute(
                """UPDATE alert_log SET odds_t3=?, t3_recorded_at=?, slippage_pct=?, steam_within_3m=?
                   WHERE alert_key=?""",
                (
                    odds_t3,
                    now.isoformat(),
                    round(slip, 3),
                    steam,
                    rec["alert_key"],
                ),
            )
            updated += 1
        c.commit()

    return {"updated": updated, "skipped": skipped, "pending_before": len(rows)}


def slippage_summary() -> dict[str, Any]:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM alert_log").fetchone()[0]
        with_t3 = c.execute("SELECT COUNT(*) FROM alert_log WHERE odds_t3 IS NOT NULL").fetchone()[0]
        steam = c.execute(
            "SELECT COUNT(*) FROM alert_log WHERE steam_within_3m = 1"
        ).fetchone()[0]
        rows = c.execute(
            "SELECT slippage_pct FROM alert_log WHERE slippage_pct IS NOT NULL"
        ).fetchall()

    slips = [float(r[0]) for r in rows]
    median = sorted(slips)[len(slips) // 2] if slips else None
    steam_pct = steam / with_t3 if with_t3 else None

    recommendation = None
    if steam_pct is not None and steam_pct >= 0.40:
        recommendation = (
            "Steam >40% alert entro 3 min: valuta scraping quote più frequente "
            "(cron Betfair ogni 10-15 min pre-match)"
        )

    return {
        "n_alerts": int(total),
        "n_with_t3_snapshot": int(with_t3),
        "steam_within_3m": int(steam),
        "steam_within_3m_pct": round(steam_pct * 100, 1) if steam_pct is not None else None,
        "median_slippage_pct": round(median, 2) if median is not None else None,
        "slippage_window_min": SLIPPAGE_WINDOW_MIN,
        "steam_threshold_pct": STEAM_SLIPPAGE_PCT,
        "recommendation": recommendation,
    }
