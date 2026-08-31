"""TML-Database adapter — sorgente primaria ATP con fallback Sackmann."""

from __future__ import annotations

import subprocess
import zlib
from pathlib import Path

import pandas as pd

from modules.data_update.entity_resolution import odds_match_key
from modules.lib_paths import LIB_TML, ROOT, env_or_lib

TML_REPO = "https://github.com/Tennismylife/TML-Database.git"
RAW_TML = ROOT / "data" / "raw" / "tml"

# Schema Sackmann-like (49 colonne match)
SACKMANN_MATCH_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date",
    "match_num", "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age", "loser_id", "loser_seed", "loser_entry",
    "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age", "score", "best_of",
    "round", "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points",
]


def _stable_player_id(raw: object) -> int:
    if pd.isna(raw):
        return 0
    try:
        return int(raw)
    except (ValueError, TypeError):
        return int(zlib.adler32(str(raw).encode("utf-8")) & 0x7FFFFFFF)


def _resolve_tml_dir() -> Path | None:
    custom = env_or_lib("TML_PATH", LIB_TML)
    if custom:
        return custom
    if RAW_TML.is_dir() and any(RAW_TML.glob("*.csv")):
        return RAW_TML
    return None


def sync_tml(*, clone: bool = True, pull: bool = True) -> dict:
    """Sync TML: lib/ locale → git clone/pull in data/raw/tml."""
    local = _resolve_tml_dir()
    if local and local.resolve() != RAW_TML.resolve():
        n = len(list(local.glob("*.csv")))
        return {"ok": n > 0, "source": str(local), "n_csv": n, "mode": "lib"}

    if clone and not RAW_TML.exists():
        RAW_TML.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", TML_REPO, str(RAW_TML)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc), "source": str(RAW_TML)}
    elif pull and (RAW_TML / ".git").is_dir():
        try:
            subprocess.run(
                ["git", "-C", str(RAW_TML), "pull", "--ff-only"],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

    local = _resolve_tml_dir()
    if not local:
        return {"ok": False, "error": "TML non trovato — copia in lib/TML-Database-master", "source": str(LIB_TML)}
    n = len(list(local.glob("*.csv")))
    return {"ok": n > 0, "source": str(local), "n_csv": n, "mode": "git" if (RAW_TML / ".git").is_dir() else "lib"}


def normalize_tml_frame(df: pd.DataFrame, *, source_year: int | None = None) -> pd.DataFrame:
    """Mappa colonne TML → schema Sackmann unificato."""
    out = df.copy()
    if "indoor" in out.columns:
        out = out.drop(columns=["indoor"])

    for col in SACKMANN_MATCH_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    out["winner_id"] = out["winner_id"].map(_stable_player_id)
    out["loser_id"] = out["loser_id"].map(_stable_player_id)
    out["tourney_date"] = pd.to_datetime(out["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    out["tour"] = "ATP"
    out["data_source"] = "TML"
    if source_year is not None:
        out["source_year"] = source_year
    elif "tourney_date" in out.columns:
        out["source_year"] = out["tourney_date"].dt.year

    return out[SACKMANN_MATCH_COLS + ["tour", "data_source", "source_year"]]


def load_tml_matches(*, min_year: int = 1990, max_year: int | None = None) -> pd.DataFrame:
    """Carica match ATP da TML (file annuali YYYY.csv + ongoing)."""
    data_dir = _resolve_tml_dir()
    if not data_dir:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for f in sorted(data_dir.glob("*.csv")):
        if f.name in ("ATP_Database.csv",):
            continue
        stem = f.stem
        try:
            year = int(stem)
        except ValueError:
            if stem == "ongoing_tourneys":
                year = pd.Timestamp.now().year
            else:
                continue
        if year < min_year:
            continue
        if max_year and year > max_year:
            continue
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if df.empty or "winner_name" not in df.columns:
            continue
        frames.append(normalize_tml_frame(df, source_year=year))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("tourney_date").reset_index(drop=True)


def _match_key_df(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: odds_match_key(
            pd.Timestamp(r["tourney_date"]).strftime("%Y-%m-%d"),
            str(r["winner_name"]),
            str(r["loser_name"]),
        ),
        axis=1,
    )


def merge_atp_primary_tml(
    tml: pd.DataFrame,
    sackmann: pd.DataFrame,
) -> pd.DataFrame:
    """TML primario; Sackmann riempie buchi (WTA-only rows esclusi — solo ATP)."""
    if tml.empty:
        return sackmann
    if sackmann.empty:
        return tml

    tml = tml.copy()
    sack = sackmann.copy()
    tml["_mk"] = _match_key_df(tml)
    sack["_mk"] = _match_key_df(sack)
    sack_extra = sack[~sack["_mk"].isin(set(tml["_mk"]))]
    out = pd.concat([tml, sack_extra], ignore_index=True)
    return out.drop(columns=["_mk"], errors="ignore").sort_values("tourney_date").reset_index(drop=True)
