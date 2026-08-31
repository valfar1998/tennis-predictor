"""Pre-match archive SQLite + settle pipeline."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "processed" / "our_history.sqlite"

_CREATE = """
CREATE TABLE IF NOT EXISTS matches (
    match_key TEXT PRIMARY KEY,
    date TEXT,
    player_a TEXT,
    player_b TEXT,
    surface TEXT,
    tourney TEXT,
    pick TEXT,
    action TEXT,
    probability REAL,
    odds REAL,
    ev REAL,
    kelly REAL,
    odds_source TEXT,
    p_markov REAL,
    p_elo REAL,
    p_ml REAL,
    hit INTEGER,
    clv REAL,
    beat_close INTEGER,
    saved_at TEXT,
    settled_at TEXT,
    score TEXT,
    retirement INTEGER
)
"""


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute(_CREATE)
    return c


def archive_prediction(pred: dict[str, Any]) -> None:
    rec = pred.get("recommended") or {}
    key = f"{pred.get('player_a')}|{pred.get('player_b')}|{pred.get('surface', '')}"
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO matches
            (match_key, date, player_a, player_b, surface, tourney, pick, action,
             probability, odds, ev, kelly, odds_source, p_markov, p_elo, p_ml, saved_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key,
                pred.get("date"),
                pred.get("player_a"),
                pred.get("player_b"),
                pred.get("surface"),
                pred.get("tourney"),
                rec.get("player"),
                pred.get("action"),
                rec.get("probability"),
                rec.get("odds"),
                rec.get("ev"),
                rec.get("kelly"),
                rec.get("odds_source"),
                pred.get("p_markov"),
                pred.get("p_elo"),
                pred.get("p_ml"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def load_history(limit: int = 500) -> list[dict]:
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM matches ORDER BY saved_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
