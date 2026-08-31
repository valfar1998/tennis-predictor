"""Registro permanente mapping giocatori ATP/WTA (SQLite) — ID univoci + alias."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from modules.data_update.entity_resolution import _canonical_key, _last_name, _norm_name

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "player_registry.sqlite"

_CREATE = """
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tour TEXT NOT NULL,
    sackmann_id INTEGER,
    tml_id TEXT,
    atp_official_id TEXT,
    canonical_name TEXT NOT NULL,
    name_first TEXT,
    name_last TEXT,
    ioc TEXT,
    UNIQUE(tour, sackmann_id),
    UNIQUE(tour, tml_id)
);
CREATE INDEX IF NOT EXISTS idx_players_canonical ON players(canonical_name);
CREATE INDEX IF NOT EXISTS idx_players_last ON players(name_last);

CREATE TABLE IF NOT EXISTS aliases (
    alias_norm TEXT PRIMARY KEY,
    player_id INTEGER NOT NULL,
    source TEXT,
    FOREIGN KEY(player_id) REFERENCES players(id)
);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(_CREATE)
    c.row_factory = sqlite3.Row
    return c


def _full_name(first: str, last: str) -> str:
    return f"{str(first or '').strip()} {str(last or '').strip()}".strip()


def register_player(
    *,
    tour: str,
    canonical_name: str,
    sackmann_id: int | None = None,
    tml_id: str | None = None,
    atp_official_id: str | None = None,
    name_first: str | None = None,
    name_last: str | None = None,
    ioc: str | None = None,
    aliases: list[str] | None = None,
    source: str = "manual",
) -> int:
    """Inserisce o aggiorna giocatore; ritorna player_id interno."""
    tour = tour.upper()
    with _conn() as c:
        row = None
        if sackmann_id is not None:
            row = c.execute(
                "SELECT id FROM players WHERE tour=? AND sackmann_id=?",
                (tour, sackmann_id),
            ).fetchone()
        if row is None and tml_id:
            row = c.execute(
                "SELECT id FROM players WHERE tour=? AND tml_id=?",
                (tour, tml_id),
            ).fetchone()

        if row:
            pid = int(row["id"])
            c.execute(
                """UPDATE players SET canonical_name=?, name_first=?, name_last=?, ioc=?,
                   tml_id=COALESCE(?, tml_id), atp_official_id=COALESCE(?, atp_official_id)
                   WHERE id=?""",
                (canonical_name, name_first, name_last, ioc, tml_id, atp_official_id, pid),
            )
        else:
            cur = c.execute(
                """INSERT INTO players
                (tour, sackmann_id, tml_id, atp_official_id, canonical_name, name_first, name_last, ioc)
                VALUES (?,?,?,?,?,?,?,?)""",
                (tour, sackmann_id, tml_id, atp_official_id, canonical_name, name_first, name_last, ioc),
            )
            pid = int(cur.lastrowid)

        alias_set = {_norm_name(canonical_name), _canonical_key(canonical_name)}
        if name_last:
            alias_set.add(_norm_name(name_last))
        for a in aliases or []:
            alias_set.add(_norm_name(a))
            alias_set.add(_canonical_key(a))

        for al in alias_set:
            if not al:
                continue
            c.execute(
                "INSERT OR REPLACE INTO aliases (alias_norm, player_id, source) VALUES (?,?,?)",
                (al, pid, source),
            )
        c.commit()
        return pid


def lookup_player_id(name: str) -> int | None:
    """Risolve nome → player_id interno via alias registry."""
    norm = _norm_name(name)
    key = _canonical_key(name)
    with _conn() as c:
        for alias in (norm, key):
            if not alias:
                continue
            row = c.execute(
                "SELECT player_id FROM aliases WHERE alias_norm=?", (alias,)
            ).fetchone()
            if row:
                return int(row["player_id"])
    return None


def resolve_canonical(name: str) -> str | None:
    """Ritorna canonical_name se il giocatore è nel registry."""
    pid = lookup_player_id(name)
    if pid is None:
        return None
    with _conn() as c:
        row = c.execute("SELECT canonical_name FROM players WHERE id=?", (pid,)).fetchone()
    return str(row["canonical_name"]) if row else None


def sync_sackmann_players(*, tour: str = "ATP") -> dict:
    """Importa atp_players.csv / wta_players.csv nel registry (batch, singola connessione)."""
    from modules.lib_paths import LIB_SACKMANN_ATP, LIB_SACKMANN_WTA

    tour = tour.upper()
    prefix = "atp" if tour == "ATP" else "wta"
    for base in (LIB_SACKMANN_ATP if tour == "ATP" else LIB_SACKMANN_WTA,):
        path = base / f"{prefix}_players.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, low_memory=False)
        n = 0
        with _conn() as c:
            for _, row in df.iterrows():
                pid_raw = row.get("player_id")
                if pd.isna(pid_raw):
                    continue
                sackmann_id = int(pid_raw)
                first = str(row.get("name_first") or "")
                last = str(row.get("name_last") or "")
                canon = _full_name(first, last)
                if not canon:
                    continue
                ioc = str(row.get("ioc") or "") or None

                existing = c.execute(
                    "SELECT id FROM players WHERE tour=? AND sackmann_id=?",
                    (tour, sackmann_id),
                ).fetchone()
                if existing:
                    player_id = int(existing["id"])
                    c.execute(
                        """UPDATE players SET canonical_name=?, name_first=?, name_last=?, ioc=?
                           WHERE id=?""",
                        (canon, first, last, ioc, player_id),
                    )
                else:
                    cur = c.execute(
                        """INSERT INTO players
                        (tour, sackmann_id, canonical_name, name_first, name_last, ioc)
                        VALUES (?,?,?,?,?,?)""",
                        (tour, sackmann_id, canon, first, last, ioc),
                    )
                    player_id = int(cur.lastrowid)

                alias_set = {_norm_name(canon), _canonical_key(canon)}
                if last:
                    alias_set.add(_norm_name(last))
                for al in alias_set:
                    if al:
                        c.execute(
                            "INSERT OR REPLACE INTO aliases (alias_norm, player_id, source) VALUES (?,?,?)",
                            (al, player_id, "sackmann"),
                        )
                n += 1
            c.commit()
        return {"ok": True, "tour": tour, "registered": n}
    return {"ok": False, "tour": tour, "error": "players csv non trovato"}


def sync_tml_players() -> dict:
    """Importa ATP_Database.csv (TML) e collega per nome+cognome."""
    from modules.lib_paths import LIB_TML

    path = LIB_TML / "ATP_Database.csv"
    if not path.is_file():
        return {"ok": False, "error": "ATP_Database.csv assente"}

    df = pd.read_csv(path, low_memory=False)
    linked = 0
    with _conn() as c:
        for _, row in df.iterrows():
            tml_id = str(row.get("id") or "").strip()
            player = str(row.get("player") or "").strip()
            atpname = str(row.get("atpname") or "").strip()
            if not tml_id or not player:
                continue
            parts = player.split(",")
            if len(parts) >= 2:
                last, first = parts[0].strip(), parts[1].strip()
            else:
                last, first = player, ""
            canon = _full_name(first, last)
            last_norm = _norm_name(last)

            existing = c.execute(
                "SELECT id FROM players WHERE tour='ATP' AND LOWER(name_last)=?",
                (last_norm,),
            ).fetchone()
            if existing:
                c.execute(
                    "UPDATE players SET tml_id=?, atp_official_id=? WHERE id=?",
                    (tml_id, atpname or None, int(existing["id"])),
                )
                pid = int(existing["id"])
            else:
                cur = c.execute(
                    """INSERT INTO players
                    (tour, tml_id, atp_official_id, canonical_name, name_first, name_last)
                    VALUES ('ATP',?,?,?,?,?)""",
                    (tml_id, atpname or None, canon, first, last),
                )
                pid = int(cur.lastrowid)

            for al in {_norm_name(canon), _norm_name(player), _norm_name(atpname), _canonical_key(canon)}:
                if al:
                    c.execute(
                        "INSERT OR REPLACE INTO aliases (alias_norm, player_id, source) VALUES (?,?,?)",
                        (al, pid, "tml"),
                    )
            linked += 1
        c.commit()
    return {"ok": True, "linked": linked}


def registry_stats() -> dict:
    with _conn() as c:
        n_players = c.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        n_aliases = c.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
    return {"players": int(n_players), "aliases": int(n_aliases), "path": str(DB_PATH)}
