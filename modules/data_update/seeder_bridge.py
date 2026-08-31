"""Bridge opzionale verso seeder (TennisExplorer odds + fixtures)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "data" / "raw" / "seeder"


def _db_url() -> str | None:
    return os.environ.get("SEEDER_DB_CONN_STR", "").strip() or None


def export_seeder_data() -> dict:
    """Esporta match e odds da DB seeder (SQLite/MySQL) in CSV."""
    conn_str = _db_url()
    if not conn_str:
        seeder_local = Path(os.environ.get("SEEDER_PATH", r"C:\Users\valba\Downloads\seeder-main"))
        sqlite = seeder_local / "seeder.db"
        if sqlite.exists():
            conn_str = f"sqlite:///{sqlite}"
        else:
            return {
                "ok": False,
                "error": "Imposta SEEDER_DB_CONN_STR o esegui seeder crawl prima",
            }

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        return {"ok": False, "error": "sqlalchemy non installato (pip install sqlalchemy)"}

    engine = create_engine(conn_str)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    exported = {}

    for table, fname in [("matches", "matches.csv"), ("match_odds", "odds.csv"), ("players", "players.csv")]:
        try:
            df = pd.read_sql(text(f"SELECT * FROM {table} LIMIT 500000"), engine)
            if not df.empty:
                dest = EXPORT_DIR / fname
                df.to_csv(dest, index=False)
                exported[table] = len(df)
        except Exception:
            continue

    return {"ok": bool(exported), "exported": exported, "dir": str(EXPORT_DIR)}


def load_seeder_odds() -> pd.DataFrame:
    path = EXPORT_DIR / "odds.csv"
    if not path.exists():
        result = export_seeder_data()
        if not result.get("ok"):
            return pd.DataFrame()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)
