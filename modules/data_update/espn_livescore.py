"""Risultati tennis da ESPN scoreboard API (ATP/WTA, senza API key)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "espn_results.json"
BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
FINAL_STATUSES = frozenset({"STATUS_FINAL", "STATUS_FULL_TIME"})


def _http_get(url: str) -> dict | None:
    headers = {
        "User-Agent": UA,
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.espn.com/tennis/scoreboard",
        "Origin": "https://www.espn.com",
    }
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, headers=headers, impersonate="chrome120", timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    try:
        import requests

        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _score_text(competition: dict, competitors: list[dict]) -> str | None:
    status = competition.get("status") or {}
    short = (status.get("type") or {}).get("shortDetail")
    if short and short.lower() not in ("final", "ft"):
        return str(short)
    parts: list[str] = []
    for comp in competitors:
        lines = comp.get("linescores") or []
        if not lines:
            continue
        vals = [str(ls.get("displayValue") or ls.get("value") or "").strip() for ls in lines]
        vals = [v for v in vals if v]
        if vals:
            parts.append("-".join(vals))
    if len(parts) == 2:
        return f"{parts[0]} {parts[1]}"
    return str(short) if short else None


def _parse_competition(competition: dict, *, tour: str, event_name: str, day: str) -> dict | None:
    status_name = str((competition.get("status") or {}).get("type", {}).get("name") or "")
    if status_name not in FINAL_STATUSES:
        return None
    competitors = competition.get("competitors") or []
    if len(competitors) != 2:
        return None

    names: dict[str, str] = {}
    winner: str | None = None
    for comp in competitors:
        athlete = comp.get("athlete") or {}
        name = str(athlete.get("displayName") or comp.get("displayName") or "").strip()
        if not name:
            return None
        side = str(comp.get("homeAway") or ("home" if not names else "away"))
        names[side] = name
        if comp.get("winner"):
            winner = name

    player_a = names.get("home") or names.get("away")
    player_b = names.get("away") if player_a == names.get("home") else names.get("home")
    if not player_a or not player_b or not winner:
        return None
    loser = player_b if winner == player_a else player_a
    comp_day = str(competition.get("date") or competition.get("startDate") or day)[:10]
    return {
        "player_a": player_a,
        "player_b": player_b,
        "winner": winner,
        "loser": loser,
        "score": _score_text(competition, competitors),
        "date": comp_day,
        "tour": tour.upper(),
        "tournament": event_name,
        "source": "espn",
        "status": status_name,
    }


def _parse_scoreboard(payload: dict, *, tour: str, day: str) -> list[dict]:
    rows: list[dict] = []
    for event in payload.get("events") or []:
        event_name = str(event.get("name") or event.get("shortName") or "")
        pools: list[dict] = list(event.get("competitions") or [])
        for grouping in event.get("groupings") or []:
            pools.extend(grouping.get("competitions") or [])
        for competition in pools:
            row = _parse_competition(competition, tour=tour, event_name=event_name, day=day)
            if row:
                rows.append(row)
    return rows


def _date_range(*, days: int) -> list[str]:
    today = datetime.now(timezone.utc).date()
    out: list[str] = []
    for offset in range(days + 1):
        d = today - timedelta(days=offset)
        out.append(d.strftime("%Y%m%d"))
    return out


def fetch_espn_results(
    *,
    days: int = 5,
    force: bool = False,
    max_age_hours: float = 0.5,
    tours: tuple[str, ...] = ("atp", "wta"),
) -> dict:
    """Scarica risultati finali ESPN per gli ultimi N giorni (ATP + WTA)."""
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
    dates = _date_range(days=days)

    from modules.ops_progress import log_item

    total = len(dates) * len(tours)
    step = 0
    for ymd in dates:
        iso_day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        for tour in tours:
            step += 1
            log_item(step, total, f"ESPN {tour.upper()} {iso_day}")
            url = f"{BASE}/{tour}/scoreboard?dates={ymd}"
            payload = _http_get(url)
            if not payload:
                errors.append(f"fetch_fail:{tour}:{iso_day}")
                continue
            for row in _parse_scoreboard(payload, tour=tour, day=iso_day):
                key = "|".join(
                    sorted(
                        [
                            row["player_a"].lower(),
                            row["player_b"].lower(),
                            str(row.get("date") or iso_day),
                        ]
                    )
                )
                if key in seen:
                    continue
                seen.add(key)
                if not row.get("date"):
                    row["date"] = iso_day
                matches.append(row)

    info = {
        "ok": bool(matches),
        "source": "espn",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "matches": matches,
        "days": days,
        "errors": errors or None,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load_espn_results(*, days: int = 5) -> list[dict]:
    if CACHE.is_file():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            if data.get("matches"):
                return data["matches"]
        except Exception:
            pass
    return fetch_espn_results(days=days, force=False).get("matches") or []
