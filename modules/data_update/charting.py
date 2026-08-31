"""Match Charting Project — supporto path locale + parsing stats strutturato."""

from __future__ import annotations

import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from modules.data_update.cache_policy import is_fresh
from modules.lib_paths import LIB_MCP, env_or_lib

ROOT = Path(__file__).resolve().parents[2]
CHARTING_DIR = ROOT / "data" / "raw" / "charting"
GITHUB_ZIP = "https://github.com/JeffSackmann/tennis_MatchChartingProject/archive/refs/heads/master.zip"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"


def _resolve_source() -> Path | None:
    p = env_or_lib("MCP_CHARTING_PATH", LIB_MCP)
    if p:
        return p
    if CHARTING_DIR.exists() and list(CHARTING_DIR.glob("charting-*-matches.csv")):
        return CHARTING_DIR
    return None


def sync_charting_data(*, force: bool = False, copy: bool = True) -> dict:
    """Sincronizza MCP da path locale o GitHub (CC BY-NC-SA)."""
    src = _resolve_source()
    CHARTING_DIR.mkdir(parents=True, exist_ok=True)

    if src and src.resolve() != CHARTING_DIR.resolve() and copy:
        n = 0
        for f in src.glob("charting-*.csv"):
            dst = CHARTING_DIR / f.name
            if force or not dst.exists():
                shutil.copy2(f, dst)
            n += 1
        if n:
            marker = CHARTING_DIR / ".synced"
            marker.write_text(json.dumps({"source": str(src), "n_files": n}), encoding="utf-8")
            return {"ok": True, "n_files": n, "source": str(src), "from_cache": False}

    marker = CHARTING_DIR / ".synced"
    if not force and is_fresh(marker, max_age_hours=336):
        n = len(list(CHARTING_DIR.glob("charting-*.csv")))
        if n:
            return {"ok": True, "n_files": n, "from_cache": True}

    try:
        req = Request(GITHUB_ZIP, headers={"User-Agent": UA})
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith(".csv") and "charting-" in name.lower():
                    zf.extract(name, CHARTING_DIR)
        marker.write_text(json.dumps({"synced": True}), encoding="utf-8")
        n = len(list(CHARTING_DIR.glob("charting-*.csv")))
        return {"ok": True, "n_files": n, "from_cache": False}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def load_charting_overview(*, tour: str = "m") -> pd.DataFrame:
    """Carica stats Overview (serve/return aggregati per match)."""
    sync_charting_data()
    path = CHARTING_DIR / f"charting-{tour}-stats-Overview.csv"
    if not path.exists():
        src = _resolve_source()
        if src:
            path = src / path.name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_charting_matches(*, tour: str = "m") -> pd.DataFrame:
    sync_charting_data()
    path = CHARTING_DIR / f"charting-{tour}-matches.csv"
    if not path.exists():
        src = _resolve_source()
        if src:
            path = src / path.name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def player_pressure_profile(player_name: str, *, tour: str = "m") -> dict | None:
    """Profilo clutch da MCP: BP save, tiebreak/deuce serve win rate."""
    sync_charting_data()
    src = _resolve_source()
    base = CHARTING_DIR if (CHARTING_DIR / f"charting-{tour}-stats-KeyPointsServe.csv").exists() else src
    if not base:
        return None

    kp_path = base / f"charting-{tour}-stats-KeyPointsServe.csv"
    sb_path = base / f"charting-{tour}-stats-SvBreakTotal.csv"
    if not kp_path.exists():
        return None

    kp = pd.read_csv(kp_path, low_memory=False)
    last = str(player_name).strip().split()[-1].lower()
    rows = kp[kp["player"].str.lower().str.contains(last, na=False)]
    if rows.empty:
        return None

    def _pt_rate(sub: pd.DataFrame) -> float | None:
        pts = sub["pts"].sum()
        if pts <= 0:
            return None
        return float(sub["pts_won"].sum() / pts)

    bp = rows[rows["row"] == "BP"]
    tb = rows[rows["row"].isin(["4", "5"])]
    deuce = rows[rows["row"].isin(["d", "a"])]

    profile: dict = {"n_matches": int(rows["match_id"].nunique())}
    if not bp.empty:
        profile["bp_save_rate"] = _pt_rate(bp)
    if not tb.empty:
        profile["tb_clutch"] = _pt_rate(tb)
    elif not deuce.empty:
        profile["tb_clutch"] = _pt_rate(deuce)

    if sb_path.exists():
        sb = pd.read_csv(sb_path, low_memory=False)
        sb_p = sb[sb["player"].str.lower().str.contains(last, na=False)]
        pts = sb_p["pts"].sum()
        if pts > 0:
            profile["serve_pt_won"] = float(sb_p["pts_won"].sum() / pts)

    if profile.get("bp_save_rate") is None and profile.get("serve_pt_won"):
        profile["bp_save_rate"] = profile["serve_pt_won"]
    return profile if len(profile) > 1 else None


def player_serve_profile(player_name: str, *, tour: str = "m") -> dict | None:
    """Profilo servizio medio da MCP per un giocatore."""
    overview = load_charting_overview(tour=tour)
    if overview.empty:
        return None
    key = str(player_name).strip().lower()
    rows = overview[overview["player"].str.lower().str.contains(key.split()[-1], na=False)]
    if rows.empty:
        return None
    total = rows[rows["set"] == "Total"] if "set" in rows.columns else rows
    if total.empty:
        total = rows
    agg = total.mean(numeric_only=True)
    svpt = agg.get("serve_pts", 0)
    if svpt <= 0:
        return None
    return {
        "p_first_serve_in": agg.get("first_in", 0) / svpt,
        "p_first_serve_won": agg.get("first_won", 0) / max(agg.get("first_in", 1), 1),
        "p_second_serve_won": agg.get("second_won", 0) / max(agg.get("second_in", 1), 1),
        "ace_rate": agg.get("aces", 0) / svpt,
        "df_rate": agg.get("dfs", 0) / svpt,
        "n_matches": len(total),
    }


def load_charting_summary() -> pd.DataFrame:
    return load_charting_overview()
