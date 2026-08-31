"""Percorsi dati esterni in lib/ (repo locale)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"

# Cartelle in lib/ (nome cartella = nome repo scaricato)
LIB_SACKMANN_ARCHIVE = LIB / "tennis-sackmann-archive-main"
LIB_SACKMANN_ATP = LIB_SACKMANN_ARCHIVE / "atp"
LIB_SACKMANN_WTA = LIB_SACKMANN_ARCHIVE / "wta"
LIB_ATP = LIB / "tennis_atp-master"
LIB_MCP = LIB / "tennis_MatchChartingProject-master"
LIB_INFOTENNIS = LIB / "infotennis-main"
LIB_SEEDER = LIB / "seeder-main"
LIB_TML = LIB / "TML-Database-master"
LIB_TENNISGNN = LIB / "tennisgnn_predictions-main"


def resolve_path(path: str | Path | None) -> Path | None:
    """Risolve path assoluto o relativo alla root del progetto."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p if p.exists() else None


def env_or_lib(env_var: str, lib_path: Path) -> Path | None:
    """Priorità: variabile .env → cartella lib/ default."""
    env = os.environ.get(env_var, "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    return lib_path if lib_path.exists() else None


def sackmann_archive_path() -> Path | None:
    env = os.environ.get("SACKMANN_ARCHIVE_PATH", "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    return LIB_SACKMANN_ARCHIVE if LIB_SACKMANN_ARCHIVE.is_dir() else None


def seeder_db_path() -> Path:
    return LIB_SEEDER / "seeder.db"


def resolve_sqlite_url(conn_str: str) -> str:
    """Risolve sqlite:///path relativi rispetto alla root del progetto."""
    prefix = "sqlite:///"
    if conn_str.startswith(prefix):
        rel = conn_str[len(prefix) :]
        p = Path(rel)
        if not p.is_absolute() and not (len(rel) > 1 and rel[1] == ":"):
            p = (ROOT / rel).resolve()
            return f"{prefix}{p.as_posix()}"
    return conn_str


def seeder_db_url() -> str:
    custom = os.environ.get("SEEDER_DB_CONN_STR", "").strip()
    if custom:
        return resolve_sqlite_url(custom)
    return f"sqlite:///{seeder_db_path().resolve().as_posix()}"
