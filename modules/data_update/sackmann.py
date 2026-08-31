"""Sincronizzazione dati Jeff Sackmann (ATP/WTA tour-level + players + rankings)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from modules.lib_paths import LIB_ATP, LIB_SACKMANN_ATP, LIB_SACKMANN_WTA, sackmann_archive_path

ROOT = Path(__file__).resolve().parents[2]
RAW_ATP = ROOT / "data" / "raw" / "atp"
RAW_WTA = ROOT / "data" / "raw" / "wta"
PROCESSED = ROOT / "data" / "processed"

# JeffSackmann/tennis_* possono risultare 404 su GitHub (rimossi o non raggiungibili).
# Mirror pubblico con gli stessi CSV (snapshot giu 2026):
SACKMANN_ARCHIVE = "https://github.com/Aneeshers/tennis-sackmann-archive.git"

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


def _tour_ready(dest: Path, prefix: str) -> bool:
    return (dest / f"{prefix}_players.csv").is_file() and any(dest.glob(f"{prefix}_matches_*.csv"))


def clone_sackmann_tour(*, tour: str, dest: Path | None = None) -> dict:
    """Scarica CSV Sackmann: repo originale JeffSackmann, fallback mirror archive."""
    import subprocess

    tour = tour.lower()
    if tour not in _TOUR_CFG:
        raise ValueError(f"Tour non supportato: {tour}")
    cfg = _TOUR_CFG[tour]
    prefix = cfg["prefix"]
    dest = dest or cfg["raw_dir"]

    if _tour_ready(dest, prefix):
        return {"ok": True, "source": "local", "path": str(dest)}

    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", cfg["repo"], str(dest)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if _tour_ready(dest, prefix):
            return {"ok": True, "source": "jeffsackmann", "path": str(dest)}
    except Exception as exc:
        primary_err = str(exc)

    tmp = dest.parent / f".sackmann_{tour}_clone"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                SACKMANN_ARCHIVE, str(tmp),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
        subprocess.run(
            ["git", "-C", str(tmp), "sparse-checkout", "set", tour],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        src = tmp / tour
        if not _tour_ready(src, prefix):
            raise FileNotFoundError(f"mirror senza CSV in {tour}/")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
        return {
            "ok": True,
            "source": "archive_mirror",
            "path": str(dest),
            "mirror": SACKMANN_ARCHIVE,
            "note": "JeffSackmann/tennis_{} non disponibile; usato mirror".format(tour),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "primary_error": locals().get("primary_err"),
            "hint": (
                f"Scarica manualmente da {SACKMANN_ARCHIVE} (cartella {tour}/) "
                f"e imposta {cfg['env_var']}"
            ),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _path_from_env(value: str) -> Path | None:
    if not value.strip():
        return None
    p = Path(value.strip())
    if not p.is_absolute():
        p = ROOT / p
    return p if p.exists() else None


def _resolve_source(tour: str) -> Path:
    cfg = _TOUR_CFG[tour.lower()]
    prefix = cfg["prefix"]
    tour_key = tour.lower()

    archive = sackmann_archive_path()
    if archive:
        sub = archive / tour_key
        if _tour_ready(sub, prefix):
            return sub

    env = os.environ.get(cfg["env_var"], "").strip()
    if env:
        p = _path_from_env(env)
        if p and _tour_ready(p, prefix):
            return p

    lib_tour = LIB_SACKMANN_ATP if tour_key == "atp" else LIB_SACKMANN_WTA
    if _tour_ready(lib_tour, prefix):
        return lib_tour

    if tour_key == "atp" and _tour_ready(LIB_ATP, prefix):
        return LIB_ATP

    local = cfg["raw_dir"]
    if local.exists() and any(local.glob(f"{prefix}_matches_*.csv")):
        return local
    raise FileNotFoundError(
        f"Dati Sackmann {tour.upper()} non trovati. "
        f"Imposta SACKMANN_ARCHIVE_PATH, {cfg['env_var']}, copia in lib/ o in {local}/"
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
        data_dir = raw_dir
    else:
        data_dir = src
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
    """Carica match ATP: TML primario + fallback Sackmann."""
    from modules.data_update.tml import load_tml_matches, merge_atp_primary_tml

    tml = load_tml_matches(min_year=min_year, max_year=max_year)
    try:
        sack = load_sackmann_matches(tour="atp", min_year=min_year, max_year=max_year)
    except FileNotFoundError:
        sack = pd.DataFrame()
    return merge_atp_primary_tml(tml, sack)


def ensure_sackmann_atp(*, clone: bool = True) -> dict:
    """Assicura dati ATP in data/raw/atp."""
    if not clone:
        return {"ok": _tour_ready(RAW_ATP, "atp"), "path": str(RAW_ATP)}
    return clone_sackmann_tour(tour="atp")


def ensure_sackmann_wta(*, clone: bool = True) -> dict:
    """Assicura dati WTA in data/raw/wta (mirror archive se JeffSackmann 404)."""
    if not clone:
        return {"ok": _tour_ready(RAW_WTA, "wta"), "path": str(RAW_WTA)}
    return clone_sackmann_tour(tour="wta")


def load_wta_matches(*, min_year: int = 1990, max_year: int | None = None) -> pd.DataFrame:
    """Carica match WTA Sackmann."""
    return load_sackmann_matches(tour="wta", min_year=min_year, max_year=max_year)
