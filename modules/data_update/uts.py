"""Scraping CPI (Court Pace Index) da Ultimate Tennis Statistics (gratuito)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "raw" / "uts_cpi.json"
# Pagina pubblica con indici superficie/torneo
URL = "https://www.ultimatetennisstatistics.com/tournamentPace"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"


def fetch_uts_cpi(*, force: bool = False) -> dict:
    """Scarica CPI tornei da UTS."""
    if not force and is_fresh(CACHE_PATH, max_age_hours=168):
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {"ok": True, "n_tournaments": len(data.get("tournaments", [])), "from_cache": True}

    try:
        resp = requests.get(URL, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        tournaments: list[dict] = []

        for tr in soup.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            name = cells[0]
            cpi = _parse_float(cells[1])
            if cpi is None:
                continue
            tournaments.append({"name": name, "cpi": cpi, "surface": cells[2] if len(cells) > 2 else None})

        payload = {"tournaments": tournaments, "source": URL}
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "n_tournaments": len(tournaments), "from_cache": False}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _parse_float(s: str) -> float | None:
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)) or "0")
    except ValueError:
        return None


def load_cpi_index() -> dict[str, float]:
    if not CACHE_PATH.exists():
        fetch_uts_cpi()
    if not CACHE_PATH.exists():
        return {}
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {t["name"].lower(): t["cpi"] for t in data.get("tournaments", [])}


def lookup_cpi(tourney_name: str) -> float | None:
    idx = load_cpi_index()
    key = str(tourney_name or "").strip().lower()
    if key in idx:
        return idx[key]
    for name, cpi in idx.items():
        if name in key or key in name:
            return cpi
    return None


def cpi_serve_adjustment(cpi: float | None) -> float:
    """CPI alto = campo veloce → leggero boost al server."""
    if cpi is None:
        return 0.0
    # CPI tipico ~0.9-1.1, neutro ~1.0
    return max(-0.02, min(0.02, (cpi - 1.0) * 0.05))
