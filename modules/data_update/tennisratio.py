"""TennisRatio.com — skill ratings ATP/WTA (scraping leggero)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "raw" / "tennisratio_rankings.json"
BASE_URL = "https://www.tennisratio.com"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"


def fetch_tennisratio_rankings(*, force: bool = False) -> dict:
    """Scarica skill ratings dalla homepage TennisRatio (ATP/WTA)."""
    if not force and is_fresh(CACHE_PATH, max_age_hours=24):
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {"ok": True, "n_players": len(data.get("players", [])), "from_cache": True}

    try:
        resp = requests.get(BASE_URL, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        players: list[dict] = []

        # Cerca blocchi player con skill ratings (struttura homepage)
        for block in soup.find_all(["div", "article", "section"]):
            text = block.get_text(" ", strip=True)
            if "Service Games Won" not in text and "skills rating" not in text.lower():
                continue
            name_el = block.find(["h2", "h3", "h4", "strong"])
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            if len(name) < 3 or name.lower() in ("atp", "wta", "compare players"):
                continue
            skills = {}
            for metric in (
                "Service Games Won", "1st Serve Pts Won", "2nd Serve Pts Won",
                "Break Pts Saved", "Return Games Won", "Break Pts Converted",
                "Dominance Ratio", "Match Efficiency",
            ):
                m = re.search(rf"{re.escape(metric)}\s*(\d+)", text)
                if m:
                    skills[metric] = int(m.group(1))
            if skills:
                players.append({"name": name, "skills": skills, "tour": _detect_tour(text)})

        # Deduplica per nome
        seen = set()
        unique = []
        for p in players:
            key = p["name"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        payload = {"players": unique, "source": BASE_URL}
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "n_players": len(unique), "from_cache": False}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _detect_tour(text: str) -> str:
    return "WTA" if "wta" in text.lower() else "ATP"


def load_skill_index() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        fetch_tennisratio_rankings()
    if not CACHE_PATH.exists():
        return {}
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {p["name"].lower(): p for p in data.get("players", [])}


def lookup_player_skills(player_name: str) -> dict | None:
    idx = load_skill_index()
    key = str(player_name).strip().lower()
    if key in idx:
        return idx[key].get("skills")
    last = key.split()[-1]
    for name, row in idx.items():
        if last in name:
            return row.get("skills")
    return None


def skill_serve_adjustment(skills: dict | None) -> float:
    """Correzione P(serve) da skill ratings TennisRatio (0-100)."""
    if not skills:
        return 0.0
    sg = skills.get("Service Games Won", 50)
    s1 = skills.get("1st Serve Pts Won", 50)
    composite = 0.6 * sg + 0.4 * s1
    return max(-0.03, min(0.03, (composite - 70) * 0.001))
