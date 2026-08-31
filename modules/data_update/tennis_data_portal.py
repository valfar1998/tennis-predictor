"""Integrazione portale tennis-data.co.uk — quote storiche ATP/WTA + tornei.

Fonte: http://www.tennis-data.co.uk/ (Joseph Buchdahl)
- Stagioni ATP: /{year}/{year}.xlsx
- Torneo ATP: /{year}/{slug}.csv
- Torneo WTA: /{year}w/{slug}.csv
"""

from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from modules.data_update.cache_policy import is_fresh
from modules.data_update.tennis_data_catalog import PRIORITY_SLUGS, csv_stem

ROOT = Path(__file__).resolve().parents[2]
ODDS_ATP = ROOT / "data" / "raw" / "odds"
ODDS_WTA = ROOT / "data" / "raw" / "odds" / "wta"
META_PATH = ROOT / "data" / "raw" / "tennis_data_sync.json"

BASE_URL = "http://www.tennis-data.co.uk"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0; +tennis-data.co.uk)"

ATP_YEARS = list(range(2013, 2027))


def _download_bytes(url: str, *, timeout: int = 60) -> bytes | None:
    try:
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _save_xlsx_as_csv(data: bytes, dest: Path) -> bool:
    try:
        df = pd.read_excel(BytesIO(data), engine="openpyxl")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        return True
    except Exception:
        return False


def _save_csv(data: bytes, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def download_atp_seasons(*, years: list[int] | None = None, force: bool = False) -> dict:
    """Scarica file stagione ATP (.xlsx → CSV locale)."""
    years = years or ATP_YEARS
    downloaded = skipped = failed = 0

    for year in years:
        dest = ODDS_ATP / f"{year}.csv"
        if not force and is_fresh(dest, max_age_hours=168):
            skipped += 1
            continue

        url = f"{BASE_URL}/{year}/{year}.xlsx"
        data = _download_bytes(url)
        if data and _save_xlsx_as_csv(data, dest):
            downloaded += 1
            continue

        zip_url = f"{BASE_URL}/{year}/xls/{year}.zip"
        zip_data = _download_bytes(zip_url)
        ok = False
        if zip_data:
            try:
                with zipfile.ZipFile(BytesIO(zip_data)) as zf:
                    for name in zf.namelist():
                        if name.endswith((".xlsx", ".xls")):
                            ok = _save_xlsx_as_csv(zf.read(name), dest)
                            if ok:
                                break
            except Exception:
                ok = False
        if ok:
            downloaded += 1
        else:
            failed += 1

    return {"downloaded": downloaded, "skipped": skipped, "failed": failed, "tour": "ATP"}


def _tournament_url(*, year: int, slug: str, tour: str) -> str:
    stem = csv_stem(slug)
    prefix = f"{year}w" if tour.upper() == "WTA" else str(year)
    return f"{BASE_URL}/{prefix}/{stem}.csv"


def download_tournament(
    slug: str,
    *,
    year: int,
    tour: str = "ATP",
    force: bool = False,
) -> bool:
    """Scarica CSV singolo torneo."""
    dest_dir = ODDS_WTA if tour.upper() == "WTA" else ODDS_ATP / "tournaments"
    dest = dest_dir / f"{slug}_{year}.csv"
    if not force and is_fresh(dest, max_age_hours=72):
        return dest.is_file()

    url = _tournament_url(year=year, slug=slug, tour=tour)
    data = _download_bytes(url, timeout=45)
    if not data or len(data) < 100:
        return False
    return _save_csv(data, dest)


def download_priority_tournaments(
    *,
    years: list[int] | None = None,
    slugs: tuple[str, ...] | None = None,
    force: bool = False,
) -> dict:
    """Scarica Grand Slam + Masters ATP e WTA."""
    from datetime import datetime

    y_max = datetime.now().year
    years = years or list(range(y_max - 2, y_max + 1))
    slugs = slugs or PRIORITY_SLUGS
    ok = fail = 0

    for slug in slugs:
        for year in years:
            for tour in ("ATP", "WTA"):
                if download_tournament(slug, year=year, tour=tour, force=force):
                    ok += 1
                else:
                    fail += 1

    return {"ok": ok, "failed": fail, "slugs": len(slugs), "years": len(years)}


def sync_tennis_data_portal(*, force: bool = False) -> dict:
    """Sync completo tennis-data.co.uk (stagioni + tornei prioritari)."""
    atp = download_atp_seasons(force=force)
    tourn = download_priority_tournaments(force=force)
    info = {
        "source": BASE_URL,
        "livescore": "http://livescore.tennis-data.co.uk/",
        "atp_seasons": atp,
        "tournaments": tourn,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _read_csv_dir(directory: Path, *, tour_label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not directory.is_dir():
        return pd.DataFrame()
    for f in sorted(directory.glob("*.csv")):
        try:
            df = pd.read_csv(f, low_memory=False)
            m = re.search(r"(\d{4})", f.stem)
            if m:
                df["odds_year"] = int(m.group(1))
            df["odds_tour"] = tour_label
            df["odds_file"] = f.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_odds_atp() -> pd.DataFrame:
    """Quote ATP: stagioni + tornei."""
    frames = []
    for f in sorted(ODDS_ATP.glob("*.csv")):
        try:
            df = pd.read_csv(f, low_memory=False)
            m = re.search(r"(\d{4})", f.stem)
            if m:
                df["odds_year"] = int(m.group(1))
            df["odds_tour"] = "ATP"
            frames.append(df)
        except Exception:
            continue
    frames.append(_read_csv_dir(ODDS_ATP / "tournaments", tour_label="ATP"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return out.drop_duplicates(subset=["Date", "Winner", "Loser"], keep="last") if all(
        c in out.columns for c in ("Date", "Winner", "Loser")
    ) else out


def load_odds_wta() -> pd.DataFrame:
    return _read_csv_dir(ODDS_WTA, tour_label="WTA")


def load_odds_all() -> pd.DataFrame:
    atp = load_odds_atp()
    wta = load_odds_wta()
    if atp.empty and wta.empty:
        return pd.DataFrame()
    if atp.empty:
        return wta
    if wta.empty:
        return atp
    return pd.concat([atp, wta], ignore_index=True)


def lookup_pinnacle_odds(
    player_a: str,
    player_b: str,
    *,
    date: str | None = None,
    tour: str = "ATP",
) -> dict | None:
    """Cerca PSW/PSL Pinnacle per match nel cache tennis-data."""
    from modules.data_update.entity_resolution import _last_name, odds_match_key

    df = load_odds_wta() if tour.upper() == "WTA" else load_odds_atp()
    if df.empty:
        return None

    day = str(date or "")[:10]
    key = odds_match_key(day, player_a, player_b)

    w_col = next((c for c in ("Winner", "winner") if c in df.columns), None)
    l_col = next((c for c in ("Loser", "loser") if c in df.columns), None)
    if not w_col or not l_col:
        return None

    df = df.copy()
    date_col = next((c for c in ("Date", "date") if c in df.columns), None)
    if not date_col:
        return None

    df["_key"] = df.apply(
        lambda r: odds_match_key(
            pd.Timestamp(r[date_col]).strftime("%Y-%m-%d"),
            str(r[w_col]),
            str(r[l_col]),
        ),
        axis=1,
    )

    hit = df[df["_key"] == key]
    if hit.empty:
        hit = df[df["_key"] == odds_match_key(day, player_b, player_a)]
    if hit.empty:
        return None

    row = hit.iloc[-1]
    psw = row.get("PSW") or row.get("AvgW")
    psl = row.get("PSL") or row.get("AvgL")
    if pd.isna(psw) or pd.isna(psl):
        return None

    winner = str(row[w_col])
    if _last_name(winner) == _last_name(player_a):
        return {"a": float(psw), "b": float(psl), "source": "tennis-data.co.uk"}
    return {"a": float(psl), "b": float(psw), "source": "tennis-data.co.uk"}
