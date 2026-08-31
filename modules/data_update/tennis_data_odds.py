"""Download quote storiche da tennis-data.co.uk (gratuito)."""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
ODDS_DIR = ROOT / "data" / "raw" / "odds"

BASE_URL = "http://www.tennis-data.co.uk"
ATP_YEARS = list(range(2013, 2026))

UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"


def _download_bytes(url: str, timeout: int = 45) -> bytes | None:
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


def download_tennis_data_odds(*, force: bool = False) -> dict:
    """Scarica file stagione ATP da tennis-data.co.uk (formato .xlsx)."""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = 0

    for year in ATP_YEARS:
        dest = ODDS_DIR / f"{year}.csv"
        if not force and is_fresh(dest, max_age_hours=168):
            skipped += 1
            continue

        # Formato ufficiale: /{year}/{year}.xlsx (file stagione intera)
        url = f"{BASE_URL}/{year}/{year}.xlsx"
        data = _download_bytes(url)
        if data and _save_xlsx_as_csv(data, dest):
            downloaded += 1
            continue

        # Fallback: zip stagione da alldata.php
        zip_url = f"{BASE_URL}/{year}/xls/{year}.zip"
        zip_data = _download_bytes(zip_url)
        if zip_data:
            try:
                with zipfile.ZipFile(BytesIO(zip_data)) as zf:
                    for name in zf.namelist():
                        if name.endswith((".xlsx", ".xls")):
                            if _save_xlsx_as_csv(zf.read(name), dest):
                                downloaded += 1
                                break
                    else:
                        failed += 1
                if dest.exists():
                    continue
            except Exception:
                pass

        failed += 1

    return {"downloaded": downloaded, "skipped": skipped, "failed": failed, "dir": str(ODDS_DIR)}


def load_odds_history() -> pd.DataFrame:
    """Carica tutte le quote storiche ATP disponibili."""
    frames: list[pd.DataFrame] = []
    for f in sorted(ODDS_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(f, low_memory=False)
            m = re.search(r"(\d{4})", f.stem)
            if m:
                df["odds_year"] = int(m.group(1))
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
