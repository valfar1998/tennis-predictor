"""Scraping rating Elo da Tennis Abstract (gratuito, uso personale)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "raw" / "tennis_abstract_elo.json"
URL = "https://www.tennisabstract.com/reports/atp_elo_ratings.html"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"


def fetch_tennis_abstract_elo(*, force: bool = False) -> dict:
    """Scarica e parsa la tabella Elo ATP da Tennis Abstract."""
    if not force and is_fresh(CACHE_PATH, max_age_hours=48):
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {"ok": True, "n_players": len(data.get("players", [])), "from_cache": True}

    try:
        resp = requests.get(URL, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table")
        if not table:
            return {"ok": False, "error": "tabella non trovata"}

        players: list[dict] = []
        rows = table.find_all("tr")
        for tr in rows[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) < 4:
                continue
            name = cells[0]
            try:
                elo = float(re.sub(r"[^\d.]", "", cells[1]) or "0")
            except ValueError:
                continue
            players.append({
                "name": name,
                "elo_overall": elo,
                "elo_hard": _parse_optional(cells, 2),
                "elo_clay": _parse_optional(cells, 3),
                "elo_grass": _parse_optional(cells, 4),
            })

        payload = {"players": players, "source": URL}
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "n_players": len(players), "from_cache": False}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _parse_optional(cells: list, idx: int) -> float | None:
    if idx >= len(cells):
        return None
    try:
        v = float(re.sub(r"[^\d.]", "", cells[idx]) or "0")
        return v if v > 0 else None
    except ValueError:
        return None


def load_elo_index() -> dict[str, dict]:
    """Nome normalizzato → rating Elo Tennis Abstract."""
    if not CACHE_PATH.exists():
        fetch_tennis_abstract_elo()
    if not CACHE_PATH.exists():
        return {}
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    idx: dict[str, dict] = {}
    for p in data.get("players", []):
        key = str(p.get("name", "")).strip().lower()
        if key:
            idx[key] = p
    return idx


def lookup_ta_elo(player_name: str, surface: str | None = None) -> float | None:
    idx = load_elo_index()
    key = str(player_name).strip().lower()
    row = idx.get(key)
    if not row:
        for k, v in idx.items():
            if key in k or k in key:
                row = v
                break
    if not row:
        return None
    if surface:
        s = surface.lower()
        if s == "hard" and row.get("elo_hard"):
            return row["elo_hard"]
        if s == "clay" and row.get("elo_clay"):
            return row["elo_clay"]
        if s == "grass" and row.get("elo_grass"):
            return row["elo_grass"]
    return row.get("elo_overall")
