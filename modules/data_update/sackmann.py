"""Sincronizzazione dati Jeff Sackmann (ATP/WTA tour-level + players + rankings)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_ATP = ROOT / "data" / "raw" / "atp"
RAW_WTA = ROOT / "data" / "raw" / "wta"
PROCESSED = ROOT / "data" / "processed"

_TOUR_CFG = {
    "atp": {
        "env_var": "SACKMANN_ATP_PATH",
        "raw_dir": RAW_ATP,
        "prefix": "atp",
        "repo": "https://github.com/JeffSackmann/tennis_atp.git",
    },
    "wta": {
        "env_var": "SACKMANN_WTA_PATH",
        "raw_dir": RAW_WTA,
        "prefix": "wta",
        "repo": "https://github.com/JeffSackmann/tennis_wta.git",
    },
}


def _resolve_source(tour: str) -> Path:
    cfg = _TOUR_CFG[tour.lower()]
    env = os.environ.get(cfg["env_var"], "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p
    local = cfg["raw_dir"]
    prefix = cfg["prefix"]
    if local.exists() and any(local.glob(f"{prefix}_matches_*.csv")):
        return local
    raise FileNotFoundError(
        f"Dati Sackmann {tour.upper()} non trovati. "
        f"Imposta {cfg['env_var']} o copia i CSV in {local}/"
    )


def sync_sackmann_tour(*, tour: str = "atp", copy: bool = False, min_year: int = 1990) -> dict:
    """Carica o copia i CSV Sackmann tour-level in data/raw/{atp|wta}."""
    tour = tour.lower()
    if tour not in _TOUR_CFG:
        raise ValueError(f"Tour non supportato: {tour}")
    cfg = _TOUR_CFG[tour]
    prefix = cfg["prefix"]
    raw_dir = cfg["raw_dir"]

    src = _resolve_source(tour)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if copy and src.resolve() != raw_dir.resolve():
        for pattern in (f"{prefix}_matches_*.csv", f"{prefix}_players.csv", f"{prefix}_rankings_*.csv"):
            for f in src.glob(pattern):
                if "doubles" in f.name or "futures" in f.name or "qual_chall" in f.name:
                    continue
                dst = raw_dir / f.name
                if not dst.exists() or dst.stat().st_mtime < f.stat().st_mtime:
                    shutil.copy2(f, dst)

    data_dir = raw_dir if (raw_dir / f"{prefix}_players.csv").exists() else src
    match_files = sorted(
        f for f in data_dir.glob(f"{prefix}_matches_*.csv")
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
        "tour": tour.upper(),
        "source": str(data_dir),
        "n_match_files": len(match_files),
        "years": f"{min(years)}-{max(years)}" if years else "none",
        "players_file": str(data_dir / f"{prefix}_players.csv"),
    }


def sync_sackmann_atp(*, copy: bool = False, min_year: int = 1990) -> dict:
    return sync_sackmann_tour(tour="atp", copy=copy, min_year=min_year)


def sync_sackmann_wta(*, copy: bool = False, min_year: int = 1990) -> dict:
    return sync_sackmann_tour(tour="wta", copy=copy, min_year=min_year)


def load_sackmann_matches(
    *,
    tour: str = "atp",
    min_year: int = 1990,
    max_year: int | None = None,
) -> pd.DataFrame:
    """Carica match tour-level Sackmann (ATP o WTA) in un unico DataFrame."""
    tour = tour.lower()
    info = sync_sackmann_tour(tour=tour, min_year=min_year)
    data_dir = Path(info["source"])
    prefix = _TOUR_CFG[tour]["prefix"]
    frames: list[pd.DataFrame] = []

    for f in sorted(data_dir.glob(f"{prefix}_matches_*.csv")):
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
        df["tour"] = tour.upper()
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["tourney_date"] = pd.to_datetime(out["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    return out.sort_values("tourney_date").reset_index(drop=True)


def load_tour_matches(*, min_year: int = 1990, max_year: int | None = None) -> pd.DataFrame:
    """Carica match ATP (compatibilità retro)."""
    return load_sackmann_matches(tour="atp", min_year=min_year, max_year=max_year)


def ensure_sackmann_wta(*, clone: bool = True) -> dict:
    """Assicura dati WTA in data/raw/wta (clone git se assente)."""
    cfg = _TOUR_CFG["wta"]
    raw_dir = cfg["raw_dir"]
    if (raw_dir / "wta_players.csv").is_file() and any(raw_dir.glob("wta_matches_*.csv")):
        return {"ok": True, "source": "local", "path": str(raw_dir)}

    if clone:
        import subprocess

        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            if any(raw_dir.iterdir()):
                import shutil
                shutil.rmtree(raw_dir, ignore_errors=True)
                raw_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", cfg["repo"], str(raw_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            return {"ok": True, "source": "git", "path": str(raw_dir)}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "hint": f"Imposta {cfg['env_var']} con i CSV WTA"}

    return {"ok": False, "error": "dati WTA assenti"}


def load_wta_matches(*, min_year: int = 1990, max_year: int | None = None) -> pd.DataFrame:
    """Carica match WTA Sackmann."""
    return load_sackmann_matches(tour="wta", min_year=min_year, max_year=max_year)
