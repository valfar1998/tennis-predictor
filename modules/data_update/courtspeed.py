"""Court Pace Index da courtspeed.com (database pubblico 2012-2026)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "raw" / "courtspeed_cpi.json"
URL = "https://courtspeed.com/"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"

# Alias torneo per matching fuzzy
TOURNEY_ALIASES = {
    "australian open": "aus open",
    "roland garros": "roland garros",
    "french open": "roland garros",
    "us open": "us open",
    "indian wells": "indian wells",
    "miami open": "miami",
    "monte carlo": "monte carlo",
    "madrid": "madrid",
    "rome": "rome",
    "wimbledon": "wimbledon",
    "canadian open": "canada",
    "montreal": "canada",
    "toronto": "canada",
    "cincinnati": "cincinnati",
    "shanghai": "shanghai",
    "paris masters": "paris",
    "paris": "paris",
    "atp finals": "atp finals",
    "turin": "atp finals",
}


def fetch_courtspeed_cpi(*, force: bool = False) -> dict:
    if not force and is_fresh(CACHE_PATH, max_age_hours=168):
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {"ok": True, "n_tournaments": len(data.get("tournaments", [])), "from_cache": True}

    try:
        resp = requests.get(URL, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        tournaments: list[dict] = []

        for tr in soup.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            name_cell = cells[0].get_text(" ", strip=True)
            name = re.sub(r"◆", "", name_cell).strip()
            name = re.sub(r"(Hard|Clay|Grass|indoor).*$", "", name, flags=re.I).strip()
            if not name or name.lower() in ("tournament", ""):
                continue

            surface = _detect_surface(name_cell)
            years: dict[str, float] = {}
            year_headers = [th.get_text(strip=True) for th in soup.find("tr").find_all("th")] if soup.find("tr") else []
            for i, cell in enumerate(cells[1:], start=0):
                val = _parse_cpi(cell.get_text(strip=True))
                if val is not None and i < len(year_headers):
                    yr = year_headers[i].replace("'", "20") if "'" in year_headers[i] else year_headers[i]
                    if yr.isdigit() or (len(yr) == 2 and yr.isdigit()):
                        year_key = yr if len(yr) == 4 else f"20{yr}"
                        years[year_key] = val
                elif val is not None:
                    years[f"col_{i}"] = val

            latest = _latest_cpi(cells[1:])
            if latest is not None:
                tournaments.append({
                    "name": name,
                    "surface": surface,
                    "cpi_latest": latest,
                    "cpi_3yr": _parse_cpi(cells[-1].get_text(strip=True)) if len(cells) > 1 else None,
                    "years": years,
                })

        payload = {"tournaments": tournaments, "source": URL}
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "n_tournaments": len(tournaments), "from_cache": False}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _detect_surface(text: str) -> str | None:
    t = text.lower()
    for s in ("hard", "clay", "grass"):
        if s in t:
            return s.title()
    return None


def _parse_cpi(s: str) -> float | None:
    s = str(s or "").strip()
    if not s or s.upper() in ("–", "-", "COVID", ""):
        return None
    try:
        return float(re.sub(r"[^\d.]", "", s))
    except ValueError:
        return None


def _latest_cpi(cells) -> float | None:
    for cell in reversed(list(cells)):
        val = _parse_cpi(cell.get_text(strip=True) if hasattr(cell, "get_text") else str(cell))
        if val is not None:
            return val
    return None


def load_courtspeed_index() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        fetch_courtspeed_cpi()
    if not CACHE_PATH.exists():
        return _default_cpi()
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    idx = {t["name"].lower(): t for t in data.get("tournaments", [])}
    return idx or _default_cpi()


def _default_cpi() -> dict[str, dict]:
    """Fallback statico da dati courtspeed.com (2024-2026)."""
    defaults = {
        "indian wells": {"cpi_latest": 36.9, "surface": "Hard"},
        "miami": {"cpi_latest": 35.5, "surface": "Hard"},
        "monte carlo": {"cpi_latest": 29.1, "surface": "Clay"},
        "madrid": {"cpi_latest": 27.0, "surface": "Clay"},
        "rome": {"cpi_latest": 29.3, "surface": "Clay"},
        "wimbledon": {"cpi_latest": 37.0, "surface": "Grass"},
        "cincinnati": {"cpi_latest": 42.5, "surface": "Hard"},
        "us open": {"cpi_latest": 42.8, "surface": "Hard"},
        "shanghai": {"cpi_latest": 40.8, "surface": "Hard"},
        "paris": {"cpi_latest": 45.5, "surface": "Hard"},
        "atp finals": {"cpi_latest": 39.9, "surface": "Hard"},
    }
    return {k: {**v, "name": k} for k, v in defaults.items()}


def lookup_courtspeed_cpi(tourney_name: str, *, surface: str | None = None) -> float | None:
    idx = load_courtspeed_index()
    key = str(tourney_name or "").strip().lower()
    for alias, target in TOURNEY_ALIASES.items():
        if alias in key:
            key = target
            break
    if key in idx:
        return idx[key].get("cpi_latest") or idx[key].get("cpi_3yr")
    for name, row in idx.items():
        if name in key or key in name:
            if surface and row.get("surface") and surface.lower() != row["surface"].lower():
                continue
            return row.get("cpi_latest") or row.get("cpi_3yr")
    return None


def cpi_serve_adjustment(cpi: float) -> float:
    """CPI CourtSpeed: 30=lento, 45=veloce. Neutro ~37."""
    return max(-0.025, min(0.025, (cpi - 37) * 0.002))
