"""Sincronizzazione dati Jeff Sackmann (ATP tour-level + players + rankings)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_ATP = ROOT / "data" / "raw" / "atp"
PROCESSED = ROOT / "data" / "processed"


def _resolve_source() -> Path:
    env = os.environ.get("SACKMANN_ATP_PATH", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p
    local = RAW_ATP
    if local.exists() and any(local.glob("atp_matches_*.csv")):
        return local
    raise FileNotFoundError(
        "Dati Sackmann non trovati. Imposta SACKMANN_ATP_PATH o copia i CSV in data/raw/atp/"
    )


def sync_sackmann_atp(*, copy: bool = False, min_year: int = 1990) -> dict:
    """Carica o copia i CSV Sackmann ATP tour-level in data/raw/atp."""
    src = _resolve_source()
    RAW_ATP.mkdir(parents=True, exist_ok=True)

    if copy and src.resolve() != RAW_ATP.resolve():
        for pattern in ("atp_matches_*.csv", "atp_players.csv", "atp_rankings_*.csv"):
            for f in src.glob(pattern):
                if "doubles" in f.name or "futures" in f.name or "qual_chall" in f.name:
                    continue
                dst = RAW_ATP / f.name
                if not dst.exists() or dst.stat().st_mtime < f.stat().st_mtime:
                    shutil.copy2(f, dst)

    data_dir = RAW_ATP if (RAW_ATP / "atp_players.csv").exists() else src
    match_files = sorted(
        f for f in data_dir.glob("atp_matches_*.csv")
        if "doubles" not in f.name and "futures" not in f.name and "qual_chall" not in f.name
    )
    years = []
    for f in match_files:
        try:
            y = int(f.stem.split("_")[-1])
            if y >= min_year:
                years.append(y)
        except ValueError:
            continue

    return {
        "source": str(data_dir),
        "n_match_files": len(match_files),
        "years": f"{min(years)}-{max(years)}" if years else "none",
        "players_file": str(data_dir / "atp_players.csv"),
    }


def load_tour_matches(*, min_year: int = 1990, max_year: int | None = None) -> pd.DataFrame:
    """Carica tutti i match tour-level ATP in un unico DataFrame."""
    info = sync_sackmann_atp()
    data_dir = Path(info["source"])
    frames: list[pd.DataFrame] = []

    for f in sorted(data_dir.glob("atp_matches_*.csv")):
        if any(x in f.name for x in ("doubles", "futures", "qual_chall", "amateur")):
            continue
        try:
            year = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        if year < min_year:
            continue
        if max_year and year > max_year:
            continue
        df = pd.read_csv(f, low_memory=False)
        df["source_year"] = year
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["tourney_date"] = pd.to_datetime(out["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    return out.sort_values("tourney_date").reset_index(drop=True)
