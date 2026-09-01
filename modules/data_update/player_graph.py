"""Graph matching giocatori: ID ATP/WTA, nazionalità, DOB, storico incontri."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from modules.data_update.entity_resolution import _canonical_key, _last_name, _norm_name
from modules.data_update.player_registry import DB_PATH, _conn, lookup_player_id

_CREATE_GRAPH = """
CREATE TABLE IF NOT EXISTS match_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id_a INTEGER NOT NULL,
    player_id_b INTEGER NOT NULL,
    tourney_date TEXT,
    tourney_name TEXT,
    tour TEXT,
    FOREIGN KEY(player_id_a) REFERENCES players(id),
    FOREIGN KEY(player_id_b) REFERENCES players(id)
);
CREATE INDEX IF NOT EXISTS idx_edges_a ON match_edges(player_id_a, tourney_date);
CREATE INDEX IF NOT EXISTS idx_edges_b ON match_edges(player_id_b, tourney_date);
CREATE INDEX IF NOT EXISTS idx_edges_tourney ON match_edges(tourney_name, tourney_date);
"""


def _ensure_graph_schema() -> None:
    with _conn() as c:
        c.executescript(_CREATE_GRAPH)
        cols = {row[1] for row in c.execute("PRAGMA table_info(players)").fetchall()}
        if "birth_year" not in cols:
            c.execute("ALTER TABLE players ADD COLUMN birth_year INTEGER")
        if "dob" not in cols:
            c.execute("ALTER TABLE players ADD COLUMN dob TEXT")
        c.commit()


def _dob_year(dob: str | int | None) -> int | None:
    if dob is None or (isinstance(dob, float) and pd.isna(dob)):
        return None
    s = str(int(dob)) if str(dob).replace(".", "").isdigit() else str(dob).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def sync_player_biographics(players_df: pd.DataFrame, *, tour: str = "ATP") -> int:
    """Aggiorna DOB/IOC da atp_players.csv / wta_players.csv."""
    _ensure_graph_schema()
    tour = tour.upper()
    n = 0
    with _conn() as c:
        for _, row in players_df.iterrows():
            pid_raw = row.get("player_id")
            if pd.isna(pid_raw):
                continue
            sackmann_id = int(pid_raw)
            dob = row.get("dob")
            ioc = str(row.get("ioc") or "") or None
            by = _dob_year(dob)
            existing = c.execute(
                "SELECT id FROM players WHERE tour=? AND sackmann_id=?",
                (tour, sackmann_id),
            ).fetchone()
            if not existing:
                continue
            c.execute(
                "UPDATE players SET ioc=COALESCE(?, ioc), dob=?, birth_year=? WHERE id=?",
                (ioc, str(int(dob)) if pd.notna(dob) else None, by, int(existing["id"])),
            )
            n += 1
        c.commit()
    return n


def build_match_edges(matches: pd.DataFrame, *, tour: str = "ATP", limit: int | None = None) -> int:
    """Popola grafo incontri da storico Sackmann/TML."""
    _ensure_graph_schema()
    if matches is None or matches.empty:
        return 0

    m = matches.sort_values("tourney_date")
    if limit:
        m = m.tail(limit)

    inserted = 0
    with _conn() as c:
        c.execute("DELETE FROM match_edges WHERE tour=?", (tour.upper(),))
        for _, row in m.iterrows():
            try:
                wid, lid = int(row["winner_id"]), int(row["loser_id"])
            except (TypeError, ValueError):
                continue
            pa = c.execute(
                "SELECT id FROM players WHERE tour=? AND sackmann_id=?", (tour.upper(), wid)
            ).fetchone()
            pb = c.execute(
                "SELECT id FROM players WHERE tour=? AND sackmann_id=?", (tour.upper(), lid)
            ).fetchone()
            if not pa or not pb:
                continue
            dt = str(row.get("tourney_date") or "")[:10]
            tn = str(row.get("tourney_name") or "")
            c.execute(
                """INSERT INTO match_edges (player_id_a, player_id_b, tourney_date, tourney_name, tour)
                   VALUES (?,?,?,?,?)""",
                (int(pa["id"]), int(pb["id"]), dt, tn, tour.upper()),
            )
            inserted += 1
        c.commit()
    return inserted


def _candidates_by_last_name(last: str, tour: str | None = None) -> list[sqlite3.Row]:
    if not last:
        return []
    with _conn() as c:
        q = "SELECT * FROM players WHERE LOWER(name_last)=?"
        params: list = [last.lower()]
        if tour:
            q += " AND tour=?"
            params.append(tour.upper())
        return list(c.execute(q, params).fetchall())


def _resolve_by_opponent(
    name: str,
    *,
    opponent_name: str | None,
    tourney_date: str | None,
    tour: str | None,
) -> str | None:
    if not opponent_name:
        return None
    opp_id = lookup_player_id(opponent_name)
    if opp_id is None:
        opp_last = _last_name(opponent_name)
        hits = _candidates_by_last_name(opp_last, tour)
        if len(hits) == 1:
            opp_id = int(hits[0]["id"])
        else:
            return None

    date = str(tourney_date or "")[:10]
    with _conn() as c:
        rows = c.execute(
            """SELECT p.canonical_name, p.id
               FROM match_edges e
               JOIN players p ON (p.id = e.player_id_a OR p.id = e.player_id_b)
               WHERE (e.player_id_a=? OR e.player_id_b=?)
                 AND (?='' OR e.tourney_date=?)
               """,
            (opp_id, opp_id, date, date),
        ).fetchall()

    last = _last_name(name)
    cands = [r for r in rows if _last_name(r["canonical_name"]) == last]
    if len(cands) == 1:
        return str(cands[0]["canonical_name"])
    return None


def graph_resolve_player(
    name: str,
    *,
    tour: str | None = None,
    opponent_name: str | None = None,
    tourney_date: str | None = None,
    ioc: str | None = None,
) -> str | None:
    """Risolve nome via grafo (opponent + data torneo) prima del fuzzy."""
    _ensure_graph_schema()

    by_opp = _resolve_by_opponent(
        name, opponent_name=opponent_name, tourney_date=tourney_date, tour=tour
    )
    if by_opp:
        return by_opp

    last = _last_name(name)
    cands = _candidates_by_last_name(last, tour)
    if len(cands) == 1:
        return str(cands[0]["canonical_name"])

    if ioc and len(cands) > 1:
        ioc_hits = [r for r in cands if str(r["ioc"] or "").upper() == ioc.upper()]
        if len(ioc_hits) == 1:
            return str(ioc_hits[0]["canonical_name"])

    key = _canonical_key(name)
    if not key:
        return None
    with _conn() as c:
        rows = c.execute(
            """SELECT p.canonical_name, p.birth_year, p.ioc
               FROM aliases a JOIN players p ON p.id=a.player_id
               WHERE a.alias_norm LIKE ?""",
            (f"%{key.split()[0]}%",),
        ).fetchall()
    if len(rows) == 1:
        return str(rows[0]["canonical_name"])
    return None


def graph_stats() -> dict:
    _ensure_graph_schema()
    with _conn() as c:
        n_edges = c.execute("SELECT COUNT(*) FROM match_edges").fetchone()[0]
        n_bio = c.execute("SELECT COUNT(*) FROM players WHERE birth_year IS NOT NULL").fetchone()[0]
    return {"match_edges": int(n_edges), "players_with_dob": int(n_bio)}
