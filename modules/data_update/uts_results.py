"""Risultati torneo da Ultimate Tennis Statistics (bootgrid API)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "uts_results.json"
BASE = "https://www.ultimatetennisstatistics.com"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"


def _get_table(
    path: str,
    *,
    params: dict | None = None,
) -> list[dict]:
    try:
        resp = requests.get(
            f"{BASE}{path}",
            params=params or {},
            headers={"Accept": "application/json", "User-Agent": UA},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("rows") or []
    except Exception:
        return []


def _final_match_from_event(row: dict) -> dict | None:
    """Estrae finale torneo (winner vs runner-up) — utile per pick sulla finale."""
    winner = (row.get("winner") or {}).get("name")
    runner = (row.get("runnerUp") or {}).get("name")
    if not winner or not runner:
        return None
    return {
        "player_a": winner,
        "player_b": runner,
        "winner": winner,
        "loser": runner,
        "score": row.get("score"),
        "date": str(row.get("date") or "")[:10],
        "tour": "ATP",
        "tournament": row.get("name"),
        "source": "uts_final",
        "season": row.get("season"),
    }


def fetch_uts_results(
    *,
    days: int = 30,
    force: bool = False,
    max_age_hours: float = 2.0,
    row_count: int = 100,
) -> dict:
    """Scarica risultati UTS (finali torneo + metadati). Match-level live via ESPN/TA charting."""
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

    from modules.ops_progress import log_item

    log_item(1, 2, "UTS tournamentEventsTable")
    rows = _get_table(
        "/tournamentEventsTable",
        params={"current": 1, "rowCount": row_count, "sort[date]": "desc"},
    )
    for row in rows:
        day = str(row.get("date") or "")[:10]
        try:
            if day and datetime.fromisoformat(day).date() < cutoff:
                continue
        except ValueError:
            pass
        parsed = _final_match_from_event(row)
        if not parsed:
            continue
        key = "|".join(sorted([parsed["player_a"].lower(), parsed["player_b"].lower(), parsed["date"]]))
        if key in seen:
            continue
        seen.add(key)
        matches.append(parsed)

    # Grand Slam correnti (spesso non ancora in UTS DB — best effort)
    log_item(2, 2, "UTS search US Open / Wimbledon")
    for phrase in ("US Open", "Wimbledon", "Roland Garros", "Australian Open"):
        extra = _get_table(
            "/tournamentEventsTable",
            params={"current": 1, "rowCount": 20, "searchPhrase": phrase},
        )
        for row in extra:
            parsed = _final_match_from_event(row)
            if not parsed:
                continue
            key = "|".join(sorted([parsed["player_a"].lower(), parsed["player_b"].lower(), parsed["date"]]))
            if key in seen:
                continue
            seen.add(key)
            matches.append(parsed)

    info = {
        "ok": bool(matches),
        "source": "uts",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "matches": matches,
        "days": days,
        "note": "UTS fornisce finali torneo; per match live usare ESPN/TA charting/RapidAPI",
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load_uts_results(*, days: int = 30) -> list[dict]:
    if CACHE.is_file():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            if data.get("matches"):
                return data["matches"]
        except Exception:
            pass
    return fetch_uts_results(days=days, force=False).get("matches") or []
