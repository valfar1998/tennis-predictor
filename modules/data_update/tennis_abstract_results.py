"""Risultati live/post-match da Tennis Abstract (charting punto-per-punto, Grand Slam)."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "tennis_abstract_results.json"
BASE = "https://www.tennisabstract.com"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"
CHARTING_LINK_RE = re.compile(
    r"charting/(20\d{6})-([MW])-([^\"']+?)\.html",
    re.IGNORECASE,
)
RESULT_RE = re.compile(
    r"([\w][\w\s\.'\u00c0-\u024f-]+?)\s+d\.\s+([\w][\w\s\.'\u00c0-\u024f-]+?)\s+([\d\-\(\)\s]+)",
    re.UNICODE,
)


def _http_get(url: str) -> str | None:
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=45)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def _underscore_name(raw: str) -> str:
    return str(raw or "").replace("_", " ").strip()


def _parse_charting_link(path: str) -> dict | None:
    m = CHARTING_LINK_RE.search(path)
    if not m or path.endswith("meta.html"):
        return None
    ymd, tour_flag, rest = m.group(1), m.group(2).upper(), m.group(3)
    parts = rest.split("-")
    if len(parts) < 3:
        return None
    player_b = _underscore_name(parts[-1])
    player_a = _underscore_name(parts[-2])
    tournament = "_".join(parts[:-2])
    day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    return {
        "path": path,
        "date": day,
        "tour": "WTA" if tour_flag == "W" else "ATP",
        "tournament": tournament.replace("_", " "),
        "player_a_guess": player_a,
        "player_b_guess": player_b,
    }


def _discover_charting_links(*, days: int = 7) -> list[dict]:
    html = _http_get(f"{BASE}/")
    sources = [html] if html else []
    # meta.html elenca migliaia di match chartati (Grand Slam inclusi)
    meta = _http_get(f"{BASE}/charting/meta.html")
    if meta:
        sources.append(meta)

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    found: list[dict] = []
    seen: set[str] = set()
    for blob in sources:
        if not blob:
            continue
        for match in CHARTING_LINK_RE.finditer(blob):
            full = f"charting/{match.group(1)}-{match.group(2)}-{match.group(3)}.html"
            if full in seen:
                continue
            day = f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
            try:
                if datetime.fromisoformat(day).date() < cutoff:
                    continue
            except ValueError:
                continue
            seen.add(full)
            meta_row = _parse_charting_link(full)
            if meta_row:
                found.append(meta_row)
    return found


def _parse_tourney_page(html: str, *, slug: str, tour: str) -> list[dict]:
    """Parser best-effort per tourney.cgi (quando disponibile)."""
    rows: list[dict] = []
    soup = BeautifulSoup(html, "lxml")
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        text = tr.get_text(" ", strip=True)
        if " d. " not in text:
            continue
        links = tr.find_all("a", href=re.compile(r"player\.cgi"))
        if len(links) < 2:
            continue
        winner = links[0].get_text(" ", strip=True)
        loser = links[1].get_text(" ", strip=True)
        score = None
        m = re.search(r"(\d[\d\-\(\)\s]+)", text)
        if m:
            score = m.group(1).strip()
        rows.append(
            {
                "player_a": winner,
                "player_b": loser,
                "winner": winner,
                "loser": loser,
                "score": score,
                "date": None,
                "tour": tour,
                "tournament": slug,
                "source": "tennis_abstract_tourney",
            }
        )
    return rows


def _fetch_charting_match(meta: dict, *, page_cache: dict[str, dict]) -> dict | None:
    path = meta["path"]
    if path in page_cache:
        return page_cache[path]

    url = f"{BASE}/{path}"
    html = _http_get(url)
    if not html:
        return None
    m = RESULT_RE.search(html)
    if not m:
        return None
    winner, loser, score = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    row = {
        "player_a": winner,
        "player_b": loser,
        "winner": winner,
        "loser": loser,
        "score": score,
        "date": meta["date"],
        "tour": meta["tour"],
        "tournament": meta.get("tournament"),
        "source": "tennis_abstract_charting",
        "charting_path": path,
    }
    page_cache[path] = row
    time.sleep(0.08)
    return row


def fetch_tennis_abstract_results(
    *,
    days: int = 7,
    force: bool = False,
    max_age_hours: float = 1.0,
) -> dict:
    """Charting MCP + pagine torneo Tennis Abstract (Slam / tornei recenti)."""
    if not force and is_fresh(CACHE, max_age_hours=max_age_hours) and CACHE.is_file():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if cached.get("matches") is not None:
                return cached
        except Exception:
            pass

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    matches: list[dict] = []
    seen: set[str] = set()
    page_cache: dict[str, dict] = {}
    errors: list[str] = []

    from modules.ops_progress import log_item

    links = _discover_charting_links(days=days)
    slam_kw = ("us_open", "wimbledon", "roland_garros", "australian_open")
    links.sort(
        key=lambda m: (
            0 if any(k in str(m.get("tournament", "")).lower().replace(" ", "_") for k in slam_kw) else 1,
            m.get("date") or "",
        )
    )
    recent = links[:200]
    total = max(len(recent), 1) + 4
    step = 0

    for meta in recent:
        step += 1
        log_item(step, total, f"TA charting {meta['date']} {meta.get('player_a_guess')}")
        row = _fetch_charting_match(meta, page_cache=page_cache)
        if not row:
            continue
        key = "|".join(sorted([row["player_a"].lower(), row["player_b"].lower(), row["date"]]))
        if key in seen:
            continue
        seen.add(key)
        matches.append(row)

    year = datetime.now(timezone.utc).year
    slugs = [
        (f"{year}US_Open", "ATP"),
        (f"{year}US_Open_W", "WTA"),
        (f"{year}Wimbledon", "ATP"),
        (f"{year}Wimbledon_W", "WTA"),
    ]
    for slug, tour in slugs:
        step += 1
        log_item(step, total, f"TA tourney {slug}")
        html = _http_get(f"{BASE}/cgi-bin/tourney.cgi?t={slug}")
        if not html or "IndexError" in html or len(html) < 1000:
            errors.append(f"tourney_fail:{slug}")
            continue
        for row in _parse_tourney_page(html, slug=slug, tour=tour):
            day = row.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            row["date"] = day
            key = "|".join(sorted([row["player_a"].lower(), row["player_b"].lower(), day]))
            if key in seen:
                continue
            seen.add(key)
            matches.append(row)

    info = {
        "ok": bool(matches),
        "source": "tennis_abstract",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "matches": matches,
        "days": days,
        "errors": errors or None,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load_tennis_abstract_results(*, days: int = 7) -> list[dict]:
    if CACHE.is_file():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            if data.get("matches"):
                return data["matches"]
        except Exception:
            pass
    return fetch_tennis_abstract_results(days=days, force=False).get("matches") or []
