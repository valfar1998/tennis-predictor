"""Pre-match archive SQLite + settle pipeline (Sackmann results)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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
    tour TEXT,
    pick TEXT,
    action TEXT,
    probability REAL,
    odds REAL,
    ev REAL,
    ev_pct REAL,
    kelly REAL,
    odds_source TEXT,
    p_markov REAL,
    p_elo REAL,
    p_ml REAL,
    playability REAL,
    playability_band TEXT,
    moneyway_vol_pct REAL,
    dropping_pct REAL,
    dropping_aligned INTEGER,
    hit INTEGER,
    clv REAL,
    beat_close INTEGER,
    saved_at TEXT,
    settled_at TEXT,
    score TEXT,
    winner TEXT,
    retirement INTEGER
)
"""

_EXTRA_COLS = (
    ("tour", "TEXT"),
    ("ev_pct", "REAL"),
    ("playability", "REAL"),
    ("playability_band", "TEXT"),
    ("moneyway_vol_pct", "REAL"),
    ("dropping_pct", "REAL"),
    ("dropping_aligned", "INTEGER"),
    ("winner", "TEXT"),
)


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute(_CREATE)
    cols = {row[1] for row in c.execute("PRAGMA table_info(matches)")}
    for name, typ in _EXTRA_COLS:
        if name not in cols:
            c.execute(f"ALTER TABLE matches ADD COLUMN {name} {typ}")
    c.commit()
    return c


def _match_key(pred: dict[str, Any]) -> str:
    return "|".join(
        str(pred.get(k) or "")
        for k in ("player_a", "player_b", "surface", "date")
    )


def _signals(pred: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    sig = pred.get("market_signals") or {}
    mw = sig.get("volume_pct_pick")
    drop = sig.get("drop_pct")
    aligned = sig.get("aligned_with_pick")
    if aligned is None:
        return mw, drop, None
    return mw, drop, 1 if aligned else 0


def archive_prediction(pred: dict[str, Any]) -> None:
    rec = pred.get("recommended") or {}
    mw, drop, aligned = _signals(pred)
    key = _match_key(pred)
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO matches
            (match_key, date, player_a, player_b, surface, tourney, tour, pick, action,
             probability, odds, ev, ev_pct, kelly, odds_source, p_markov, p_elo, p_ml,
             playability, playability_band, moneyway_vol_pct, dropping_pct, dropping_aligned,
             clv, beat_close, saved_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key,
                str(pred.get("date") or "")[:10],
                pred.get("player_a"),
                pred.get("player_b"),
                pred.get("surface"),
                pred.get("tourney"),
                pred.get("tour"),
                rec.get("player"),
                pred.get("action"),
                rec.get("probability"),
                rec.get("odds"),
                rec.get("ev"),
                rec.get("ev_pct"),
                rec.get("kelly"),
                pred.get("odds_source"),
                pred.get("p_markov"),
                pred.get("p_elo"),
                pred.get("p_ml"),
                pred.get("playability"),
                pred.get("playability_band"),
                mw,
                drop,
                aligned,
                pred.get("clv") or rec.get("clv"),
                int(rec.get("beat_close")) if rec.get("beat_close") is not None else None,
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


def history_summary() -> dict[str, Any]:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        settled = c.execute("SELECT COUNT(*) FROM matches WHERE hit IS NOT NULL").fetchone()[0]
        hits = c.execute("SELECT COUNT(*) FROM matches WHERE hit = 1").fetchone()[0]
        pending = c.execute("SELECT COUNT(*) FROM matches WHERE hit IS NULL").fetchone()[0]
    return {
        "n_total": int(total),
        "n_settled": int(settled),
        "n_pending": int(pending),
        "n_hits": int(hits),
        "hit_rate": round(hits / settled, 4) if settled else None,
        "path": str(DB),
    }


def _load_recent_sackmann(*, days: int = 14) -> pd.DataFrame:
    from modules.data_update.sackmann import load_sackmann_matches

    cutoff = datetime.now() - timedelta(days=days)
    frames: list[pd.DataFrame] = []
    for tour in ("atp", "wta"):
        try:
            df = load_sackmann_matches(tour=tour, min_year=cutoff.year - 1)
            if not df.empty:
                frames.append(df)
        except FileNotFoundError:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["tourney_date"] = pd.to_datetime(out["tourney_date"], errors="coerce")
    return out[out["tourney_date"] >= cutoff]


def _names_match(a: str, b: str, x: str, y: str) -> bool:
    from modules.data_update.entity_resolution import _last_name

    la, lb, lx, ly = _last_name(a), _last_name(b), _last_name(x), _last_name(y)
    if not all((la, lb, lx, ly)):
        return False
    return (la == lx and lb == ly) or (la == ly and lb == lx)


def _pick_hit(pick: str, player_a: str, player_b: str, winner: str) -> int:
    from modules.data_update.entity_resolution import _last_name

    lp = _last_name(pick)
    lw = _last_name(winner)
    if not lp or not lw:
        return 0
    return 1 if lp == lw else 0


def settle_from_sackmann(*, days: int = 14) -> dict[str, Any]:
    """Chiude pick pendenti usando risultati Sackmann ATP/WTA."""
    results = _load_recent_sackmann(days=days)
    if results.empty:
        return {"settled": 0, "reason": "no_sackmann_results"}

    now = datetime.now(timezone.utc).isoformat()
    settled = 0
    with _conn() as c:
        c.row_factory = sqlite3.Row
        pending = c.execute(
            "SELECT * FROM matches WHERE hit IS NULL AND action = 'bet'"
        ).fetchall()

        for rec in pending:
            rec = dict(rec)
            day = str(rec.get("date") or "")[:10]
            if not day:
                continue
            pa, pb = str(rec["player_a"]), str(rec["player_b"])
            hit_row = None
            for _, row in results.iterrows():
                rday = pd.Timestamp(row["tourney_date"]).strftime("%Y-%m-%d")
                if abs((pd.Timestamp(rday) - pd.Timestamp(day)).days) > 2:
                    continue
                wname = str(row.get("winner_name") or "")
                lname = str(row.get("loser_name") or "")
                if not _names_match(pa, pb, wname, lname):
                    continue
                hit_row = (wname, str(row.get("score") or ""))
                break
            if not hit_row:
                continue
            winner, score = hit_row
            pick = str(rec.get("pick") or "")
            hit = _pick_hit(pick, pa, pb, winner)
            c.execute(
                """UPDATE matches SET hit=?, winner=?, score=?, settled_at=?
                   WHERE match_key=?""",
                (hit, winner, score, now, rec["match_key"]),
            )
            settled += 1
        c.commit()

    summary = history_summary()
    summary["settled"] = settled
    return summary


def settle_pending(*, learn: bool = True) -> dict[str, Any]:
    out = settle_from_sackmann()
    if learn:
        try:
            from modules.advisor.online_learn import learn_from_settled

            out["online_learn"] = learn_from_settled()
        except Exception as exc:
            out["online_learn_error"] = str(exc)
    return out
