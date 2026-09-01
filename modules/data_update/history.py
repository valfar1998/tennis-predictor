"""Pre-match archive SQLite + settle pipeline (TML → Betfair → ESPN → RapidAPI → TA → UTS → FlashScore → tennis-data)."""

from __future__ import annotations

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
    ("close_source", "TEXT"),
    ("close_odds_a", "REAL"),
    ("close_odds_b", "REAL"),
    ("settle_source", "TEXT"),
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
             clv, beat_close, close_source, close_odds_a, close_odds_b, saved_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                pred.get("close_source") or rec.get("close_source"),
                rec.get("close_a") or (pred.get("close_odds") or {}).get("a"),
                rec.get("close_b") or (pred.get("close_odds") or {}).get("b"),
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


def _names_match(a: str, b: str, x: str, y: str) -> bool:
    from modules.data_update.entity_resolution import _last_name

    la, lb, lx, ly = _last_name(a), _last_name(b), _last_name(x), _last_name(y)
    if not all((la, lb, lx, ly)):
        return False
    return (la == lx and lb == ly) or (la == ly and lb == lx)


def _pick_hit(pick: str, player_a: str, player_b: str, winner: str) -> int:
    from modules.data_update.entity_resolution import player_side_match

    if player_side_match(pick, winner):
        return 1
    return 0


def settle_from_results(*, days: int = 14) -> dict[str, Any]:
    """Chiude pick pendenti con cascade risultati (TML → Sackmann → Betfair → FlashScore → tennis-data)."""
    from modules.data_update.match_results import ResultProviders, resolve_match_result
    from modules.ops_progress import log_item, pct

    prov = ResultProviders(days=days)
    now = datetime.now(timezone.utc).isoformat()
    settled = 0
    by_source: dict[str, int] = {}

    with _conn() as c:
        c.row_factory = sqlite3.Row
        pending = c.execute(
            "SELECT * FROM matches WHERE hit IS NULL AND action = 'bet'"
        ).fetchall()
        n_pending = len(pending)
        every = max(1, n_pending // 10) if n_pending else 1

        for i, rec in enumerate(pending, 1):
            rec = dict(rec)
            day = str(rec.get("date") or "")[:10]
            if not day:
                if i == 1 or i == n_pending or i % every == 0:
                    log_item(i, n_pending, "skip: data mancante")
                continue
            pa, pb = str(rec["player_a"]), str(rec["player_b"])
            hit = resolve_match_result(
                pa,
                pb,
                date=day,
                tour=str(rec.get("tour") or ""),
                tourney=str(rec.get("tourney") or ""),
                providers=prov,
            )
            if not hit:
                if i == 1 or i == n_pending or i % every == 0:
                    log_item(i, n_pending, f"in attesa: {pa} vs {pb}")
                continue
            pick = str(rec.get("pick") or "")
            hit_val = _pick_hit(pick, pa, pb, hit.winner)
            c.execute(
                """UPDATE matches SET hit=?, winner=?, score=?, settled_at=?, settle_source=?
                   WHERE match_key=?""",
                (hit_val, hit.winner, hit.score, now, hit.source, rec["match_key"]),
            )
            settled += 1
            by_source[hit.source] = by_source.get(hit.source, 0) + 1
            log_item(i, n_pending, f"chiusa [{hit.source}]: {pa} vs {pb} -> hit={hit_val}")
        c.commit()

    summary = history_summary()
    summary["settled"] = settled
    summary["settle_by_source"] = by_source
    summary["settle_providers"] = prov.stats
    if n_pending:
        print(
            f"  settle riepilogo: {settled}/{n_pending} ({pct(settled, n_pending)}%) pick chiuse",
            flush=True,
        )
    return summary


def settle_from_sackmann(*, days: int = 14) -> dict[str, Any]:
    """Backward compat — delega a settle_from_results."""
    return settle_from_results(days=days)


def refresh_clv_close(*, days: int = 14) -> dict[str, Any]:
    """Aggiorna CLV su pick pendenti con quote di chiusura a cascata."""
    from modules.advisor.clv_live import clv_vs_close, resolve_close_odds

    updated = 0
    with _conn() as c:
        c.row_factory = sqlite3.Row
        pending = c.execute(
            """SELECT * FROM matches
               WHERE hit IS NULL AND action = 'bet'
               AND (clv IS NULL OR close_source IS NULL)"""
        ).fetchall()
        for rec in pending:
            rec = dict(rec)
            pa, pb = str(rec["player_a"]), str(rec["player_b"])
            day = str(rec.get("date") or "")[:10]
            close = resolve_close_odds(pa, pb, date=day, tour=str(rec.get("tour") or "ATP"))
            if not close:
                continue
            pick = str(rec.get("pick") or "")
            side = "A" if _last_name(pick) == _last_name(pa) else "B"
            info = clv_vs_close(
                pick_side=side,
                odds_bet=float(rec["odds"]),
                close_a=close.get("a"),
                close_b=close.get("b"),
                source=str(close.get("source") or "close"),
            )
            if info.get("clv") is None:
                continue
            c.execute(
                """UPDATE matches SET clv=?, beat_close=?, close_source=?, close_odds_a=?, close_odds_b=?
                   WHERE match_key=?""",
                (
                    info["clv"],
                    int(info["beat_close"]) if info.get("beat_close") is not None else None,
                    info.get("close_source"),
                    close.get("a"),
                    close.get("b"),
                    rec["match_key"],
                ),
            )
            updated += 1
        c.commit()
    return {"clv_refreshed": updated}


def _last_name(name: str) -> str:
    from modules.data_update.entity_resolution import _last_name as ln

    return ln(name)


def settle_pending(*, learn: bool = True) -> dict[str, Any]:
    from modules.ops_progress import OpProgress, log_done

    prog = OpProgress(11 if learn else 10, label="settle")
    prog.next("Refresh CLV close...")
    out = refresh_clv_close()
    try:
        from modules.data_update.tml import sync_tml

        prog.next("Sync TML (git pull)...")
        out["tml_sync"] = sync_tml(clone=False, pull=True)
    except Exception as exc:
        out["tml_sync_error"] = str(exc)
    try:
        from modules.data_update.flashscore import fetch_flashscore_results

        prog.next("Sync FlashScore risultati...")
        out["flashscore_sync"] = fetch_flashscore_results(force=False)
    except Exception as exc:
        out["flashscore_sync_error"] = str(exc)
    try:
        from modules.data_update.betfair import fetch_betfair_settled_results, login_configured

        prog.next("Sync Betfair settled...")
        if login_configured():
            out["betfair_settled_sync"] = fetch_betfair_settled_results(days=14, force=False)
        else:
            print("  Betfair settled skip: credenziali assenti", flush=True)
    except Exception as exc:
        out["betfair_settled_sync_error"] = str(exc)
    try:
        from modules.data_update.espn_livescore import fetch_espn_results

        prog.next("Sync ESPN risultati...")
        out["espn_sync"] = fetch_espn_results(days=5, force=False)
    except Exception as exc:
        out["espn_sync_error"] = str(exc)
    try:
        from modules.data_update.sofascore_livescore import fetch_sofascore_results

        prog.next("Sync SofaScore risultati...")
        out["sofascore_sync"] = fetch_sofascore_results(days=5, force=False)
    except Exception as exc:
        out["sofascore_sync_error"] = str(exc)
    try:
        from modules.data_update.rapidapi_tennis import fetch_rapidapi_results
        from modules.data_update.rapidapi_usage import format_usage_line

        prog.next("Sync RapidAPI tennis...")
        out["rapidapi_sync"] = fetch_rapidapi_results(days=5, force=False)
        usage = (out["rapidapi_sync"] or {}).get("rapidapi_usage")
        if usage:
            print(f"  {format_usage_line(usage)}", flush=True)
    except Exception as exc:
        out["rapidapi_sync_error"] = str(exc)
    try:
        from modules.data_update.tennis_abstract_results import fetch_tennis_abstract_results

        prog.next("Sync Tennis Abstract charting...")
        out["tennis_abstract_sync"] = fetch_tennis_abstract_results(days=7, force=False)
    except Exception as exc:
        out["tennis_abstract_sync_error"] = str(exc)
    try:
        from modules.data_update.uts_results import fetch_uts_results

        prog.next("Sync UTS risultati...")
        out["uts_sync"] = fetch_uts_results(days=30, force=False)
    except Exception as exc:
        out["uts_sync_error"] = str(exc)
    prog.next("Chiudi pick pendenti (cascade)...")
    out.update(settle_from_results())
    if learn:
        prog.next("Online learn...")
        try:
            from modules.advisor.online_learn import learn_from_settled

            out["online_learn"] = learn_from_settled()
        except Exception as exc:
            out["online_learn_error"] = str(exc)
        try:
            from modules.advisor.validation_freeze import maybe_auto_unfreeze

            auto = maybe_auto_unfreeze()
            if auto:
                out["validation_freeze_completed"] = auto
        except Exception:
            pass
    log_done("settle_pending completato")
    return out
