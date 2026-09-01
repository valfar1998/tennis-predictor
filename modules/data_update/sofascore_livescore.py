"""Risultati tennis da SofaScore API (curl_cffi per Cloudflare)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "sofascore_results.json"
BASE = "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _http_get(url: str) -> dict | None:
    headers = {"User-Agent": UA, "Accept": "application/json"}
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, headers=headers, impersonate="chrome120", timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _parse_event(event: dict, *, day: str) -> dict | None:
    status = str((event.get("status") or {}).get("type") or "")
    if status not in ("finished", "closed"):
        return None
    winner_code = event.get("winnerCode")
    if winner_code not in (1, 2):
        return None
    home = str((event.get("homeTeam") or {}).get("name") or "").strip()
    away = str((event.get("awayTeam") or {}).get("name") or "").strip()
    if not home or not away:
        return None
    winner = home if winner_code == 1 else away
    loser = away if winner == home else home
    home_score = event.get("homeScore") or {}
    away_score = event.get("awayScore") or {}
    score = None
    if home_score or away_score:
        score = f"{home_score.get('display', home_score.get('current'))}-{away_score.get('display', away_score.get('current'))}"
    return {
        "player_a": home,
        "player_b": away,
        "winner": winner,
        "loser": loser,
        "score": score,
        "date": day,
        "source": "sofascore",
        "status": status,
    }


def fetch_sofascore_results(
    *,
    days: int = 5,
    force: bool = False,
    max_age_hours: float = 0.5,
) -> dict:
    """Scarica risultati SofaScore per gli ultimi N giorni."""
    if not force and is_fresh(CACHE, max_age_hours=max_age_hours) and CACHE.is_file():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if cached.get("matches") is not None:
                return cached
        except Exception:
            pass

    matches: list[dict] = []
    seen: set[str] = set()
    errors: list[str] = []
    today = datetime.now(timezone.utc).date()

    from modules.ops_progress import log_item

    for offset in range(days + 1):
        day = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        log_item(offset + 1, days + 1, f"SofaScore {day}")
        payload = _http_get(f"{BASE}/{day}")
        if not payload:
            errors.append(f"fetch_fail:{day}")
            continue
        for event in payload.get("events") or []:
            row = _parse_event(event, day=day)
            if not row:
                continue
            key = "|".join(sorted([row["player_a"].lower(), row["player_b"].lower(), day]))
            if key in seen:
                continue
            seen.add(key)
            matches.append(row)

    info = {
        "ok": bool(matches),
        "source": "sofascore",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "matches": matches,
        "days": days,
        "errors": errors or None,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load_sofascore_results(*, days: int = 5) -> list[dict]:
    if CACHE.is_file():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            if data.get("matches"):
                return data["matches"]
        except Exception:
            pass
    return fetch_sofascore_results(days=days, force=False).get("matches") or []
