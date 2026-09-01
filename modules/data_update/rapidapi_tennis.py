"""Risultati tennis via RapidAPI (Tennis API ATP/WTA/ITF + SofaScore opzionale)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from modules.data_update.cache_policy import is_fresh
from modules.data_update.rapidapi_usage import (
    estimate_fetch_calls,
    format_usage_line,
    get_usage_summary,
    record_call,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "rapidapi_tennis_results.json"
DEFAULT_TENNIS_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
DEFAULT_SOFA_HOST = "sofascore6.p.rapidapi.com"
REQUEST_PAUSE_SEC = 0.65
FINISHED = frozenset(
    {
        "finished",
        "complete",
        "completed",
        "closed",
        "ended",
        "ft",
        "after overtime",
        "status_final",
        "status_final_short",
        "3",
    }
)


def _api_key() -> str | None:
    for key in ("RAPIDAPI_KEY", "RAPIDAPI_TENNIS_KEY", "X_RAPIDAPI_KEY"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def _headers(host: str) -> dict[str, str]:
    key = _api_key()
    if not key:
        return {}
    return {
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }


def _get_json(url: str, *, host: str, params: dict | None = None) -> Any:
    headers = _headers(host)
    if not headers:
        return None
    status_code: int | None = None
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
        status_code = resp.status_code
        record_call(host=host, url=url, status_code=status_code)
        time.sleep(REQUEST_PAUSE_SEC)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        if status_code is None:
            record_call(host=host, url=url, status_code=None)
        return None


def _tennis_fetch_enabled() -> bool:
    return (os.environ.get("RAPIDAPI_FETCH_TENNIS") or "0").strip().lower() in ("1", "true", "yes")


def _is_finished(item: dict) -> bool:
    status = item.get("status")
    if isinstance(status, dict):
        if status.get("isCancelled"):
            return False
        if status.get("isFinished"):
            return True
        st = str(status.get("type") or status.get("description") or "").lower()
        return st in FINISHED or "final" in st or "ended" in st
    st = _match_status(item)
    if not st:
        return False
    return st in FINISHED or "final" in st


def _infer_tour_from_sofa(item: dict) -> str:
    for team_key in ("homeTeam", "awayTeam"):
        gender = (item.get(team_key) or {}).get("gender")
        if gender == "F":
            return "WTA"
        if gender == "M":
            return "ATP"
    cat = str(((item.get("tournament") or {}).get("category") or {}).get("name") or "").lower()
    if "wta" in cat or "women" in cat:
        return "WTA"
    return "ATP"


def _sofa_match_date(item: dict, fallback: str) -> str:
    ts = item.get("timestamp")
    if ts:
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError):
            pass
    return _match_date(item, fallback)


def _winner_from_scores(item: dict, pa: str, pb: str) -> tuple[str | None, str | None]:
    hs = item.get("homeScore") or {}
    as_ = item.get("awayScore") or {}
    for key in ("normaltime", "current", "display"):
        hv, av = hs.get(key), as_.get(key)
        if hv is None or av is None:
            continue
        try:
            hi, ai = int(hv), int(av)
        except (TypeError, ValueError):
            continue
        if hi > ai:
            return pa, pb
        if ai > hi:
            return pb, pa
    if item.get("winnerCode") in (1, 2, "1", "2"):
        code = int(item["winnerCode"])
        return (pa, pb) if code == 1 else (pb, pa)
    return None, None


def _player_name(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        for key in ("name", "fullName", "displayName", "player_name", "shortName"):
            val = node.get(key)
            if val:
                return str(val).strip()
        first = str(node.get("firstName") or node.get("first_name") or "").strip()
        last = str(node.get("lastName") or node.get("last_name") or "").strip()
        return f"{first} {last}".strip()
    return str(node).strip()


def _match_status(item: dict) -> str:
    for key in ("status", "match_status", "state", "eventStatus"):
        val = item.get(key)
        if isinstance(val, dict):
            for sub in ("type", "name", "state", "description", "shortDetail"):
                if val.get(sub):
                    return str(val[sub]).lower()
        elif val:
            return str(val).lower()
    return ""


def _match_date(item: dict, fallback: str) -> str:
    for key in ("date", "startDate", "start_time", "commence_time", "match_date"):
        val = item.get(key)
        if val:
            return str(val)[:10]
    event = item.get("event") or {}
    for key in ("date", "startDate"):
        if event.get(key):
            return str(event[key])[:10]
    return fallback


def _extract_players(item: dict) -> tuple[str, str, str | None, str | None]:
    winner = loser = None
    for key in ("winner", "winner_name", "winnerName"):
        if item.get(key):
            winner = _player_name(item.get(key))
            break
    for key in ("loser", "loser_name", "loserName"):
        if item.get(key):
            loser = _player_name(item.get(key))
            break

    pa = pb = ""
    for a_key, b_key in (
        ("player_a", "player_b"),
        ("home", "away"),
        ("homeTeam", "awayTeam"),
        ("player1", "player2"),
        ("player_1", "player_2"),
    ):
        if item.get(a_key) and item.get(b_key):
            pa, pb = _player_name(item[a_key]), _player_name(item[b_key])
            break
    competitors = item.get("competitors") or item.get("players")
    if (not pa or not pb) and isinstance(competitors, list) and len(competitors) >= 2:
        pa, pb = _player_name(competitors[0]), _player_name(competitors[1])

    if not winner and item.get("winnerCode") in (1, 2, "1", "2"):
        code = int(item["winnerCode"])
        winner = pa if code == 1 else pb
        loser = pb if code == 1 else pa
    if not winner and item.get("winner_side") in ("home", "away", "a", "b"):
        side = str(item["winner_side"]).lower()
        winner = pa if side in ("home", "a") else pb
        loser = pb if winner == pa else pa
    if not winner and pa and pb:
        winner, loser = _winner_from_scores(item, pa, pb)

    score = item.get("score") or item.get("scores") or item.get("result")
    if isinstance(score, dict):
        score = score.get("display") or score.get("current")
    return pa, pb, winner, loser if loser else (pb if winner == pa else pa if winner == pb else None)


def _iter_payload_items(payload: Any) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "fixtures", "matches", "events", "results", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        if any(k in payload for k in ("player_a", "homeTeam", "competitors", "winner")):
            return [payload]
    return []


def _normalize_rows(items: list[dict], *, day: str, tour: str, source: str) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        if not _is_finished(item):
            continue
        pa, pb, winner, loser = _extract_players(item)
        if not pa or not pb or not winner:
            continue
        rows.append(
            {
                "player_a": pa,
                "player_b": pb,
                "winner": winner,
                "loser": loser or (pb if winner == pa else pa),
                "score": item.get("score") if isinstance(item.get("score"), str) else None,
                "date": _match_date(item, day),
                "tour": tour.upper(),
                "source": source,
                "status": _match_status(item) or "finished",
            }
        )
    return rows


def _normalize_sofa_rows(items: list[dict], *, day: str) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        if not _is_finished(item):
            continue
        pa = _player_name(item.get("homeTeam"))
        pb = _player_name(item.get("awayTeam"))
        if not pa or not pb:
            continue
        winner, loser = _winner_from_scores(item, pa, pb)
        if not winner:
            continue
        hs = item.get("homeScore") or {}
        as_ = item.get("awayScore") or {}
        score = None
        if hs.get("display") is not None and as_.get("display") is not None:
            score = f"{hs.get('display')}-{as_.get('display')}"
        rows.append(
            {
                "player_a": pa,
                "player_b": pb,
                "winner": winner,
                "loser": loser or (pb if winner == pa else pa),
                "score": score,
                "date": _sofa_match_date(item, day),
                "tour": _infer_tour_from_sofa(item),
                "source": "rapidapi_sofascore",
                "status": "finished",
                "tournament": (item.get("tournament") or {}).get("name"),
            }
        )
    return rows


def _fetch_tennis_api_fixtures(*, day: str, tour: str, host: str) -> list[dict]:
    """Calendario ATP/WTA — l'API non espone il vincitore, quindi 0 righe settle."""
    base = f"https://{host}/tennis/v2"
    urls = [
        f"{base}/{tour.lower()}/fixtures/{day}",
        f"{base}/{tour.lower()}/fixtures",
    ]
    for url in urls:
        payload = _get_json(url, host=host, params={"date": day} if url.endswith("/fixtures") else None)
        items = _iter_payload_items(payload)
        if items:
            return _normalize_rows(items, day=day, tour=tour, source="rapidapi_tennis")
    return []


def _fetch_sofa_rapidapi(*, day: str, host: str) -> list[dict]:
    """SofaScore su RapidAPI (sofascore6: /api/sofascore/v1/match/list)."""
    payload = _get_json(
        f"https://{host}/api/sofascore/v1/match/list",
        host=host,
        params={"sport_slug": "tennis", "date": day},
    )
    items = _iter_payload_items(payload)
    if items:
        return _normalize_sofa_rows(items, day=day)

    for path in (
        f"/matches/day/{day}",
        f"/sport/tennis/scheduled-events/{day}",
        f"/api/v1/sport/tennis/scheduled-events/{day}",
    ):
        payload = _get_json(f"https://{host}{path}", host=host)
        items = _iter_payload_items(payload)
        if items:
            return _normalize_sofa_rows(items, day=day)
    return []


def fetch_rapidapi_results(
    *,
    days: int = 5,
    force: bool = False,
    max_age_hours: float = 0.5,
) -> dict:
    """Scarica risultati finiti via RapidAPI (Tennis API + SofaScore opzionale)."""
    if not _api_key():
        return {
            "ok": False,
            "error": "RAPIDAPI_KEY assente nel .env",
            "matches": [],
            "n_matches": 0,
        }

    if not force and is_fresh(CACHE, max_age_hours=max_age_hours) and CACHE.is_file():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if cached.get("matches") is not None:
                cached["rapidapi_usage"] = get_usage_summary()
                return cached
        except Exception:
            pass

    tennis_host = (os.environ.get("RAPIDAPI_TENNIS_HOST") or DEFAULT_TENNIS_HOST).strip()
    sofa_host = (os.environ.get("RAPIDAPI_SOFA_HOST") or DEFAULT_SOFA_HOST).strip()
    fetch_tennis = _tennis_fetch_enabled()
    usage_before = get_usage_summary()
    est_calls = estimate_fetch_calls(
        days=days,
        include_tennis=fetch_tennis,
        include_sofa=bool(sofa_host),
    )
    if usage_before["remaining"] < est_calls:
        print(
            f"  RapidAPI quota bassa: servono ~{est_calls} chiamate, "
            f"rimangono {usage_before['remaining']} — solo SofaScore",
            flush=True,
        )
        fetch_tennis = False
        est_calls = estimate_fetch_calls(days=days, include_tennis=False, include_sofa=bool(sofa_host))

    matches: list[dict] = []
    seen: set[str] = set()
    errors: list[str] = []
    calls_made = 0
    today = datetime.now(timezone.utc).date()

    from modules.ops_progress import log_item

    per_day = (2 if fetch_tennis else 0) + (1 if sofa_host else 0)
    total = max(1, (days + 1) * per_day)
    step = 0
    for offset in range(days + 1):
        day = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        if fetch_tennis:
            for tour in ("atp", "wta"):
                step += 1
                log_item(step, total, f"RapidAPI {tour.upper()} {day}")
                try:
                    chunk = _fetch_tennis_api_fixtures(day=day, tour=tour, host=tennis_host)
                except Exception as exc:
                    errors.append(f"{tour}:{day}:{exc}")
                    chunk = []
                for row in chunk:
                    key = "|".join(sorted([row["player_a"].lower(), row["player_b"].lower(), row["date"]]))
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append(row)
        if sofa_host:
            step += 1
            log_item(step, total, f"RapidAPI SofaScore {day}")
            try:
                chunk = _fetch_sofa_rapidapi(day=day, host=sofa_host)
            except Exception as exc:
                errors.append(f"sofa:{day}:{exc}")
                chunk = []
            for row in chunk:
                key = "|".join(sorted([row["player_a"].lower(), row["player_b"].lower(), row["date"]]))
                if key in seen:
                    continue
                seen.add(key)
                matches.append(row)

    usage_after = get_usage_summary()
    calls_made = max(0, usage_after["count"] - usage_before["count"])
    print(f"  {format_usage_line(usage_after)} (+{calls_made} in questo sync)", flush=True)
    if usage_after["status"] == "warning":
        print("  RapidAPI: quota giornaliera quasi esaurita", flush=True)
    elif usage_after["status"] == "exhausted":
        print("  RapidAPI: quota giornaliera esaurita — prossimi sync useranno solo cache", flush=True)

    info = {
        "ok": bool(matches),
        "source": "rapidapi",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "matches": matches,
        "days": days,
        "tennis_host": tennis_host if fetch_tennis else None,
        "sofa_host": sofa_host or None,
        "tennis_fetch_enabled": fetch_tennis,
        "calls_this_sync": calls_made,
        "rapidapi_usage": usage_after,
        "errors": errors or None,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load_rapidapi_results(*, days: int = 5) -> list[dict]:
    if CACHE.is_file():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            if data.get("matches"):
                return data["matches"]
        except Exception:
            pass
    return fetch_rapidapi_results(days=days, force=False).get("matches") or []
