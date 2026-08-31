"""Scraping rating Elo da Tennis Abstract (ATP + WTA, gratuito, uso personale)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE_ATP = ROOT / "data" / "raw" / "tennis_abstract_elo.json"
CACHE_WTA = ROOT / "data" / "raw" / "tennis_abstract_elo_wta.json"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"

_TOUR_URL = {
    "atp": "https://www.tennisabstract.com/reports/atp_elo_ratings.html",
    "wta": "https://www.tennisabstract.com/reports/wta_elo_ratings.html",
}


def _cache_path(tour: str) -> Path:
    return CACHE_WTA if tour.lower() == "wta" else CACHE_ATP


def _parse_table(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    best: list[dict] = []

    for table in soup.find_all("table"):
        parsed: list[dict] = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True).replace("\xa0", " ") for td in tr.find_all(["td", "th"])]
            if len(cells) < 4:
                continue

            rank_idx = 0
            if cells[0] in ("Elo Rank", "Elo Rank", "Rank", "Elo\xa0Rank") or cells[0].replace("\xa0", " ") == "Elo Rank":
                continue
            if not cells[0].replace(".", "").isdigit():
                continue

            name = cells[1]
            elo_raw = cells[3] if len(cells) > 3 and re.search(r"\d{3,4}", cells[3]) else cells[2]
            try:
                elo = float(re.sub(r"[^\d.]", "", elo_raw) or "0")
            except ValueError:
                continue
            if elo < 1000 or not name or len(name) < 3:
                continue

            parsed.append({
                "name": name,
                "elo_overall": elo,
                "elo_hard": _parse_optional(cells, 6),
                "elo_clay": _parse_optional(cells, 8),
                "elo_grass": _parse_optional(cells, 10),
            })
        if len(parsed) > len(best):
            best = parsed
    return best


def fetch_tennis_abstract_elo(*, tour: str = "atp", force: bool = False) -> dict:
    """Scarica e parsa la tabella Elo Tennis Abstract (ATP o WTA)."""
    tour = tour.lower()
    if tour not in _TOUR_URL:
        raise ValueError(f"Tour non supportato: {tour}")

    cache = _cache_path(tour)
    if not force and is_fresh(cache, max_age_hours=48):
        data = json.loads(cache.read_text(encoding="utf-8"))
        return {
            "ok": True,
            "tour": tour.upper(),
            "n_players": len(data.get("players", [])),
            "from_cache": True,
        }

    try:
        url = _TOUR_URL[tour]
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        players = _parse_table(resp.text)
        if not players:
            return {"ok": False, "tour": tour.upper(), "error": "tabella non trovata"}

        payload = {"players": players, "source": url, "tour": tour.upper()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "tour": tour.upper(), "n_players": len(players), "from_cache": False}
    except Exception as exc:
        return {"ok": False, "tour": tour.upper(), "error": str(exc)}


def fetch_all_tennis_abstract_elo(*, force: bool = False) -> dict:
    atp = fetch_tennis_abstract_elo(tour="atp", force=force)
    wta = fetch_tennis_abstract_elo(tour="wta", force=force)
    return {"atp": atp, "wta": wta}


def _parse_optional(cells: list, idx: int) -> float | None:
    if idx >= len(cells):
        return None
    try:
        v = float(re.sub(r"[^\d.]", "", cells[idx]) or "0")
        return v if v >= 1000 else None
    except ValueError:
        return None


def load_elo_index(*, tour: str = "atp") -> dict[str, dict]:
    """Nome normalizzato → rating Elo Tennis Abstract."""
    tour = tour.lower()
    cache = _cache_path(tour)
    if not cache.exists():
        fetch_tennis_abstract_elo(tour=tour)
    if not cache.exists():
        return {}
    data = json.loads(cache.read_text(encoding="utf-8"))
    idx: dict[str, dict] = {}
    for p in data.get("players", []):
        key = str(p.get("name", "")).strip().lower()
        if key:
            idx[key] = p
    return idx


def lookup_ta_elo(player_name: str, surface: str | None = None, *, tour: str = "atp") -> float | None:
    resolved = str(player_name or "").strip()
    idx = load_elo_index(tour=tour)
    key = resolved.lower()
    row = idx.get(key)
    if not row:
        from modules.data_update.entity_resolution import _last_name, resolve_name

        last = _last_name(resolved)
        if last:
            for k, v in idx.items():
                if _last_name(k.replace("\xa0", " ")) == last:
                    row = v
                    break
    if not row:
        for k, v in idx.items():
            kn = k.replace("\xa0", " ")
            if key in kn or kn in key:
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
