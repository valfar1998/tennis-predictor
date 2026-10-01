"""Pre-match archive SQLite + settle pipeline (TML → Betfair → ESPN → RapidAPI → TA → UTS → FlashScore → tennis-data)."""

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
    retirement INTEGER,
    close_source TEXT,
    close_odds_a REAL,
    close_odds_b REAL,
    settle_source TEXT,
    betfair_event_id TEXT,
    betfair_market_id TEXT
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
    ("betfair_event_id", "TEXT"),
    ("betfair_market_id", "TEXT"),
    # Quote/Kelly congelate al messaggio Telegram (o al primo archive bet)
    ("odds_alert", "REAL"),
    ("kelly_alert", "REAL"),
    ("alert_frozen_at", "TEXT"),
)


def _conn() -> sqlite3.Connection:
    from modules.constants import HISTORY_BUSY_TIMEOUT_MS, HISTORY_WAL_MODE

    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=max(1.0, HISTORY_BUSY_TIMEOUT_MS / 1000.0))
    c.execute(_CREATE)
    cols = {row[1] for row in c.execute("PRAGMA table_info(matches)")}
    for name, typ in _EXTRA_COLS:
        if name not in cols:
            c.execute(f"ALTER TABLE matches ADD COLUMN {name} {typ}")
    try:
        c.execute(f"PRAGMA busy_timeout={int(HISTORY_BUSY_TIMEOUT_MS)}")
        if HISTORY_WAL_MODE:
            c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
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


def paper_eligible(pred: dict[str, Any]) -> bool:
    """Pick modello archiviabile per BCR anche se i filtri dicono no_bet."""
    if pred.get("action") == "bet":
        return False
    if pred.get("model_low_confidence"):
        pr = pred.get("players_resolved") or {}
        # Tornei inferiori: archivia paper se almeno un lato è risolto
        if not (pr.get("a") or pr.get("b")):
            return False
    from modules.advisor.advise import display_pick

    rec = display_pick(pred)
    odds = rec.get("odds")
    if not rec.get("player") or not odds or float(odds) <= 1.01:
        return False
    if rec.get("odds_real") is False:
        return False
    return True


def archive_prediction(pred: dict[str, Any]) -> None:
    from modules.advisor.advise import display_pick

    rec = pred.get("recommended") or display_pick(pred)
    action = str(pred.get("action") or "no_bet")
    mw, drop, aligned = _signals(pred)
    key = _match_key(pred)
    market_id = pred.get("betfair_market_id")
    event_id = pred.get("betfair_event_id")
    if not market_id:
        try:
            from modules.data_update.betfair import lookup_registered_market

            reg = lookup_registered_market(
                str(pred.get("player_a") or ""),
                str(pred.get("player_b") or ""),
                match_date=str(pred.get("date") or "")[:10],
                event_id=str(event_id) if event_id else None,
            )
            if reg:
                market_id = reg.get("market_id")
                if not event_id:
                    event_id = reg.get("event_id")
        except Exception:
            pass

    live_odds = rec.get("odds")
    live_kelly = rec.get("kelly_info") or rec.get("kelly")
    now = datetime.now(timezone.utc).isoformat()

    with _conn() as c:
        existing = c.execute(
            "SELECT action, hit, odds_alert, kelly_alert, alert_frozen_at FROM matches WHERE match_key=?",
            (key,),
        ).fetchone()
        if existing:
            old_action, hit = existing[0], existing[1]
            if hit is not None:
                return
            if old_action == "bet" and action != "bet":
                return
            if old_action == "shadow" and action not in ("bet", "shadow"):
                return

        odds_alert = existing[2] if existing else None
        kelly_alert = existing[3] if existing else None
        frozen_at = existing[4] if existing else None

        # Se già congelate (messaggio TG / primo bet), non aggiornare più
        if odds_alert is not None and float(odds_alert) > 1.01:
            odds_to_store = float(odds_alert)
        else:
            odds_to_store = live_odds
            if action == "bet" and live_odds is not None and float(live_odds) > 1.01:
                odds_alert = float(live_odds)
                frozen_at = frozen_at or now

        if kelly_alert is not None and float(kelly_alert) > 0:
            kelly_to_store = float(kelly_alert)
        else:
            kelly_to_store = live_kelly
            if action == "bet" and live_kelly is not None and float(live_kelly) > 0:
                kelly_alert = float(live_kelly)

        # Non scrivere CLV/chiusura all'ingresso: LTP live = quota bet, BCR finto a 0.
        c.execute(
            """INSERT OR REPLACE INTO matches
            (match_key, date, player_a, player_b, surface, tourney, tour, pick, action,
             probability, odds, ev, ev_pct, kelly, odds_source, p_markov, p_elo, p_ml,
             playability, playability_band, moneyway_vol_pct, dropping_pct, dropping_aligned,
             clv, beat_close, close_source, close_odds_a, close_odds_b,
             betfair_event_id, betfair_market_id, odds_alert, kelly_alert, alert_frozen_at, saved_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key,
                str(pred.get("date") or "")[:10],
                pred.get("player_a"),
                pred.get("player_b"),
                pred.get("surface"),
                pred.get("tourney"),
                pred.get("tour"),
                rec.get("player"),
                action,
                rec.get("probability"),
                odds_to_store,
                rec.get("ev"),
                rec.get("ev_pct"),
                kelly_to_store,
                pred.get("odds_source"),
                pred.get("p_markov"),
                pred.get("p_elo"),
                pred.get("p_ml"),
                pred.get("playability"),
                pred.get("playability_band"),
                mw,
                drop,
                aligned,
                None,
                None,
                None,
                None,
                None,
                event_id,
                market_id,
                odds_alert,
                kelly_alert,
                frozen_at,
                now,
            ),
        )


def freeze_telegram_odds(
    pred: dict[str, Any],
    *,
    odds: float | None = None,
    kelly: float | None = None,
    frozen_at: str | None = None,
) -> bool:
    """Congela quota/Kelly del messaggio Telegram sul record history (solo se assenti)."""
    from modules.advisor.advise import display_pick

    rec = pred.get("recommended") or display_pick(pred)
    try:
        odds_f = float(odds if odds is not None else rec.get("odds") or 0)
    except (TypeError, ValueError):
        odds_f = 0.0
    if odds_f <= 1.01:
        return False
    try:
        kelly_f = float(
            kelly
            if kelly is not None
            else (rec.get("kelly_info") or rec.get("kelly") or 0)
        )
    except (TypeError, ValueError):
        kelly_f = 0.0
    ts = frozen_at or datetime.now(timezone.utc).isoformat()
    key = _match_key(pred)
    with _conn() as c:
        row = c.execute(
            "SELECT odds_alert, kelly_alert FROM matches WHERE match_key=?", (key,)
        ).fetchone()
        if row is None:
            return False
        odds_alert, kelly_alert = row[0], row[1]
        already = odds_alert is not None and float(odds_alert) > 1.01
        if already:
            # Mantieni freeze; allinea solo il campo odds esposto se diverrebbe
            c.execute(
                "UPDATE matches SET odds=? WHERE match_key=? AND ABS(COALESCE(odds,0)-?)>0.0005",
                (float(odds_alert), key, float(odds_alert)),
            )
            return False
        new_kelly = (
            float(kelly_alert)
            if kelly_alert is not None and float(kelly_alert) > 0
            else (kelly_f if kelly_f > 0 else None)
        )
        c.execute(
            """UPDATE matches
               SET odds_alert=?, kelly_alert=?, odds=?,
                   kelly=COALESCE(?, kelly),
                   alert_frozen_at=COALESCE(alert_frozen_at, ?)
               WHERE match_key=?""",
            (odds_f, new_kelly, odds_f, new_kelly, ts, key),
        )
        return True


def backfill_odds_alert_from_telegram(*, limit: int = 5000) -> dict[str, Any]:
    """Riempie odds_alert da alert_log (quote messaggio TG) dove ancora assente."""
    from modules.data_update.entity_resolution import _norm_name

    updated = 0
    matched = 0
    with _conn() as c:
        c.row_factory = sqlite3.Row
        try:
            alerts = c.execute(
                """SELECT player_a, player_b, pick, odds_at_alert, sent_at, match_date, betfair_event_id
                   FROM alert_log
                   WHERE odds_at_alert IS NOT NULL AND odds_at_alert > 1.01
                   ORDER BY sent_at ASC"""
            ).fetchall()
        except sqlite3.OperationalError:
            return {"updated": 0, "matched": 0, "error": "alert_log missing"}

        bets = c.execute(
            """SELECT match_key, player_a, player_b, pick, date, betfair_event_id, odds, odds_alert
               FROM matches
               WHERE action='bet'
               ORDER BY saved_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

        by_players_date: dict[tuple[str, str, str], list] = {}
        for b in bets:
            pa = _norm_name(str(b["player_a"] or ""))
            pb = _norm_name(str(b["player_b"] or ""))
            day = str(b["date"] or "")[:10]
            if not pa or not pb or not day:
                continue
            key = (pa, pb, day)
            by_players_date.setdefault(key, []).append(b)
            by_players_date.setdefault((pb, pa, day), []).append(b)

        for a in alerts:
            pa = _norm_name(str(a["player_a"] or ""))
            pb = _norm_name(str(a["player_b"] or ""))
            day = str(a["match_date"] or "")[:10]
            if not pa or not pb or not day:
                continue
            cands = by_players_date.get((pa, pb, day)) or []
            if not cands:
                continue
            matched += 1
            odds_a = float(a["odds_at_alert"])
            pick_a = _norm_name(str(a["pick"] or ""))
            chosen = None
            for b in cands:
                if b["odds_alert"] is not None and float(b["odds_alert"]) > 1.01:
                    continue
                if pick_a and _norm_name(str(b["pick"] or "")) == pick_a:
                    chosen = b
                    break
            if chosen is None:
                for b in cands:
                    if b["odds_alert"] is None or float(b["odds_alert"] or 0) <= 1.01:
                        chosen = b
                        break
            if chosen is None:
                continue
            c.execute(
                """UPDATE matches
                   SET odds_alert=?, odds=?, alert_frozen_at=COALESCE(alert_frozen_at, ?)
                   WHERE match_key=? AND (odds_alert IS NULL OR odds_alert <= 1.01)""",
                (odds_a, odds_a, a["sent_at"], chosen["match_key"]),
            )
            if c.total_changes:
                updated += 1

        # Fallback: bet settle senza alert → congela odds attuali una tantum
        before = c.total_changes
        c.execute(
            """UPDATE matches
               SET odds_alert=odds,
                   kelly_alert=COALESCE(kelly_alert, kelly),
                   alert_frozen_at=COALESCE(alert_frozen_at, settled_at, saved_at)
               WHERE action='bet'
                 AND odds IS NOT NULL AND odds > 1.01
                 AND (odds_alert IS NULL OR odds_alert <= 1.01)"""
        )
        fallback = max(0, c.total_changes - before)

    return {"updated": updated, "matched": matched, "fallback_frozen": fallback}


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
            "SELECT * FROM matches WHERE hit IS NULL AND action IN ('bet', 'shadow', 'paper')"
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


def refresh_clv_close(*, days: int = 14, include_settled: bool = False) -> dict[str, Any]:
    """Aggiorna CLV su pick con quote di chiusura a cascata.

    Copre ``bet`` / ``shadow`` / ``paper``. Fase 2: anche se Betfair auth fallisce,
    ``resolve_close_odds`` usa prematch/fallback registry così lo shadow continua
    ad alimentare lo storico (tag quality vs fallback restano distinti).
    """
    from datetime import date, timedelta

    from modules.advisor.clv_live import clv_vs_close, resolve_close_odds

    cutoff = date.today() - timedelta(days=days - 1) if days and days > 0 else None
    updated = 0
    n_shadow = 0
    n_auth_degraded = 0
    with _conn() as c:
        c.row_factory = sqlite3.Row
        if include_settled:
            rows = c.execute(
                """SELECT * FROM matches
                   WHERE action IN ('bet', 'shadow', 'paper')"""
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT * FROM matches
                   WHERE action IN ('bet', 'shadow', 'paper')
                   AND (clv IS NULL OR close_source IS NULL
                        OR (action='shadow' AND close_source IS NULL))"""
            ).fetchall()
        for rec in rows:
            rec = dict(rec)
            day = str(rec.get("date") or "")[:10]
            if cutoff is not None:
                try:
                    if date.fromisoformat(day) < cutoff:
                        continue
                except ValueError:
                    continue
            if str(rec.get("action") or "") == "shadow":
                n_shadow += 1
            pa, pb = str(rec["player_a"]), str(rec["player_b"])
            odds_src = str(rec.get("odds_source") or "")
            try:
                close = resolve_close_odds(
                    pa,
                    pb,
                    date=day,
                    tour=str(rec.get("tour") or "ATP"),
                    betfair_event_id=rec.get("betfair_event_id"),
                    betfair_market_id=rec.get("betfair_market_id"),
                    odds_source=odds_src,
                )
            except Exception:
                close = None
            if not close:
                continue
            if close.get("auth_degraded"):
                n_auth_degraded += 1
            pick = str(rec.get("pick") or "")
            side = "A" if _last_name(pick) == _last_name(pa) else "B"
            close_pick = close.get("a") if side == "A" else close.get("b")
            odds_bet = float(rec["odds"]) if rec.get("odds") is not None else None
            src = str(close.get("source") or "").lower()
            # LTP live / snapshot / last-odds identici alla quota d'ingresso non sono una chiusura.
            if (
                odds_bet
                and close_pick
                and src in (
                    "betfair_ltp",
                    "betfair_bet_snapshot",
                    "betfair_ltp_fallback",
                    "kambi_last_odds",
                    "kambi_unibet",
                )
                and abs(float(close_pick) - odds_bet) < 0.005
            ):
                # Shadow: registra comunque fallback esplicito per non lasciare buco nello storico
                if str(rec.get("action") or "") != "shadow":
                    continue
            # Fallback last-LTP: registra comunque ma con source esplicita (BCR quality lo esclude)
            if close.get("bcr_eligible") is False and "betfair" in src and "fallback" not in src:
                close = {**close, "source": "betfair_ltp_fallback"}
            info = clv_vs_close(
                pick_side=side,
                odds_bet=odds_bet,
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
    return {
        "clv_refreshed": updated,
        "days": days,
        "include_settled": include_settled,
        "n_shadow_scanned": n_shadow,
        "n_auth_degraded_closes": n_auth_degraded,
    }


def ensure_shadow_closes(*, days: int = 21) -> dict[str, Any]:
    """Forza refresh CLV su shadow senza close (alimenta BCR sample senza bankroll)."""
    out = refresh_clv_close(days=days, include_settled=True)
    out["ok"] = True
    out["purpose"] = "shadow_bcr_feed"
    return out


def _last_name(name: str) -> str:
    from modules.data_update.entity_resolution import _last_name as ln

    return ln(name)


def retag_low_quality_betfair_closes() -> dict[str, Any]:
    """Rietichetta chiusure Betfair last-LTP (fallback) come betfair_ltp_fallback.

    Usa close_note dalla cache settled + delta |odds−close| < BCR_MIN_CLOSE_DELTA.
    """
    from modules.constants import BCR_MIN_CLOSE_DELTA
    from modules.data_update.entity_resolution import _last_name

    fallback_mids: set[str] = set()
    settled_path = ROOT / "data" / "raw" / "betfair_settled.json"
    err = None
    try:
        if settled_path.exists():
            data = json.loads(settled_path.read_text(encoding="utf-8"))
            for row in data.get("results") or []:
                note = str(row.get("close_note") or "")
                src = str(row.get("source") or "")
                if (
                    note == "last_ltp_before_close"
                    or row.get("bcr_eligible") is False
                    or "fallback" in src
                ):
                    mid = str(row.get("market_id") or "")
                    if mid:
                        fallback_mids.add(mid)
    except Exception as exc:
        err = str(exc)
        fallback_mids = set()

    updated = 0
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT match_key, pick, player_a, player_b, odds, close_odds_a, close_odds_b,
                      close_source, betfair_market_id
               FROM matches
               WHERE close_source LIKE '%betfair%'
                 AND beat_close IS NOT NULL
                 AND odds IS NOT NULL"""
        ).fetchall()
        for rec in rows:
            src = str(rec["close_source"] or "").lower()
            if "fallback" in src:
                continue
            mid = str(rec["betfair_market_id"] or "")
            retag = mid in fallback_mids
            if not retag and rec["close_odds_a"] is not None and rec["close_odds_b"] is not None:
                pick = str(rec["pick"] or "")
                pa, pb = str(rec["player_a"] or ""), str(rec["player_b"] or "")
                side = "A" if _last_name(pick) == _last_name(pa) else "B"
                try:
                    close_pick = float(rec["close_odds_a"] if side == "A" else rec["close_odds_b"])
                    odds_bet = float(rec["odds"])
                    if abs(close_pick - odds_bet) < BCR_MIN_CLOSE_DELTA:
                        retag = True
                except (TypeError, ValueError):
                    pass
            if retag:
                c.execute(
                    "UPDATE matches SET close_source=? WHERE match_key=?",
                    ("betfair_ltp_fallback", rec["match_key"]),
                )
                updated += 1
        c.commit()
    out: dict[str, Any] = {
        "retag_fallback": updated,
        "fallback_market_ids": len(fallback_mids),
    }
    if err:
        out["error"] = err
    return out


def settle_pending(*, learn: bool = True) -> dict[str, Any]:
    from modules.ops_progress import OpProgress, log_done

    prog = OpProgress(12 if learn else 11, label="settle")
    out: dict[str, Any] = {}
    # Migra schema (betfair_event_id / market_id) prima del backfill.
    try:
        _conn().close()
    except Exception:
        pass
    prog.next("Betfair market_id backfill...")
    try:
        from modules.data_update.betfair import backfill_history_market_ids, register_market_ids, load_betfair_cache

        register_market_ids(load_betfair_cache())
        out_bf = backfill_history_market_ids(days=21)
        out["betfair_market_id_backfill"] = out_bf
        print(f"  Betfair market_id backfill: {out_bf.get('updated', 0)}", flush=True)
    except Exception as exc:
        out["betfair_market_id_backfill_error"] = str(exc)
        print(f"  Betfair market_id backfill skip: {exc}", flush=True)
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
        from modules.data_update.betfair import (
            fetch_betfair_settled_results,
            fetch_betfair_odds,
            login_configured,
            snapshot_prematch_closes,
        )

        prog.next("Sync Betfair settled...")
        if login_configured():
            # Snapshot T−60/T−5/T−1 PRIMA che i mercati vadano CLOSED (senza prezzi).
            try:
                snap = snapshot_prematch_closes(days=3)
                out["betfair_prematch_snap"] = snap
            except Exception as exc:
                out["betfair_prematch_snap_error"] = str(exc)
            try:
                odds_info = fetch_betfair_odds(force=False, days=3, max_age_hours=1.0)
                out["betfair_odds_sync"] = {
                    "ok": odds_info.get("ok"),
                    "n_events": odds_info.get("n_events"),
                    "from_cache": odds_info.get("from_cache"),
                }
            except Exception as exc:
                out["betfair_odds_sync_error"] = str(exc)
            out["betfair_settled_sync"] = fetch_betfair_settled_results(days=14, force=False)
        else:
            print("  Betfair settled skip: credenziali assenti", flush=True)
    except Exception as exc:
        out["betfair_settled_sync_error"] = str(exc)
    try:
        from modules.data_update.kambi_unibet import fetch_kambi_tennis_odds, register_kambi_last_odds

        # Snapshot quote Kambi pre-match → chiusura proxy per BCR Kambi
        kambi_info = fetch_kambi_tennis_odds(force=False)
        n_reg = int(kambi_info.get("registry_updated") or 0)
        if not n_reg and kambi_info.get("events"):
            n_reg = register_kambi_last_odds(kambi_info.get("events") or [])
        out["kambi_odds_sync"] = {
            "ok": kambi_info.get("ok"),
            "n_events": kambi_info.get("n_events"),
            "from_cache": kambi_info.get("from_cache"),
            "registry_updated": n_reg,
            "error": kambi_info.get("error"),
        }
    except Exception as exc:
        out["kambi_odds_sync_error"] = str(exc)
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
    prog.next("Refresh CLV close (post-settle)...")
    try:
        out["clv_refresh"] = refresh_clv_close(days=14, include_settled=True)
        try:
            out["shadow_closes"] = ensure_shadow_closes(days=21)
        except Exception as exc:
            out["shadow_closes_error"] = str(exc)
        print(f"  CLV refreshed: {out['clv_refresh'].get('clv_refreshed', 0)}", flush=True)
    except Exception as exc:
        out["clv_refresh_error"] = str(exc)
    try:
        out["retag_closes"] = retag_low_quality_betfair_closes()
        n_retag = int((out["retag_closes"] or {}).get("retag_fallback") or 0)
        if n_retag:
            print(f"  Retag close fallback: {n_retag}", flush=True)
    except Exception as exc:
        out["retag_closes_error"] = str(exc)
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


def maintain_history(
    *,
    retain_days: int | None = None,
    vacuum: bool = True,
    archive: bool = True,
) -> dict[str, Any]:
    """Archivia settle vecchi, WAL checkpoint e VACUUM per query veloci / anti-corruzione.

    - Righe settle più vecchie di ``retain_days`` → ``our_history_archive.sqlite``
    - Unsettled / recenti restano nel DB operativo
    - ``busy_timeout`` + WAL già impostati in ``_conn``
    """
    from datetime import date, timedelta

    from modules.constants import HISTORY_RETAIN_DAYS

    days = int(retain_days if retain_days is not None else HISTORY_RETAIN_DAYS)
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    archive_path = DB.parent / "our_history_archive.sqlite"
    out: dict[str, Any] = {
        "ok": True,
        "retain_days": days,
        "cutoff": cutoff,
        "archived": 0,
        "deleted": 0,
        "wal_checkpoint": None,
        "vacuum": False,
        "archive_db": str(archive_path) if archive else None,
    }

    conn = _conn()
    try:
        conn.row_factory = sqlite3.Row
        # Settle con date vecchie (o settled_at se date assente)
        rows = conn.execute(
            """
            SELECT * FROM matches
            WHERE hit IS NOT NULL
              AND (
                    (date IS NOT NULL AND substr(date,1,10) < ?)
                 OR (date IS NULL AND settled_at IS NOT NULL AND substr(settled_at,1,10) < ?)
              )
            """,
            (cutoff, cutoff),
        ).fetchall()
        out["candidates"] = len(rows)

        if archive and rows:
            ac = sqlite3.connect(archive_path, timeout=8.0)
            try:
                ac.execute(_CREATE)
                acols = {r[1] for r in ac.execute("PRAGMA table_info(matches)")}
                for name, typ in _EXTRA_COLS:
                    if name not in acols:
                        ac.execute(f"ALTER TABLE matches ADD COLUMN {name} {typ}")
                ac.commit()
                cols = list(rows[0].keys())
                placeholders = ",".join("?" * len(cols))
                col_sql = ",".join(cols)
                for row in rows:
                    vals = [row[c] for c in cols]
                    ac.execute(
                        f"INSERT OR REPLACE INTO matches ({col_sql}) VALUES ({placeholders})",
                        vals,
                    )
                    out["archived"] += 1
                ac.commit()
            finally:
                ac.close()

            keys = [row["match_key"] for row in rows]
            for i in range(0, len(keys), 200):
                chunk = keys[i : i + 200]
                q = ",".join("?" * len(chunk))
                conn.execute(f"DELETE FROM matches WHERE match_key IN ({q})", chunk)
                out["deleted"] += len(chunk)
            conn.commit()

        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            out["journal_mode"] = mode
            ck = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            out["wal_checkpoint"] = list(ck) if ck else None
        except Exception as exc:
            out["wal_checkpoint_error"] = str(exc)

        if vacuum and out["deleted"] > 0:
            conn.commit()
            conn.execute("VACUUM")
            out["vacuum"] = True
        elif vacuum:
            out["vacuum"] = False
            out["vacuum_skipped"] = "no_rows_deleted"
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
    finally:
        conn.close()

    # Integrity check leggero
    try:
        c2 = _conn()
        integrity = c2.execute("PRAGMA integrity_check").fetchone()
        out["integrity"] = integrity[0] if integrity else None
        n = c2.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        out["n_rows"] = int(n)
        c2.close()
    except Exception as exc:
        out["integrity_error"] = str(exc)

    return out
