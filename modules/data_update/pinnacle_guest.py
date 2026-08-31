"""Pinnacle guest web API (no API key) — quote pre-match / close."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from modules.data_update.entity_resolution import _last_name

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "pinnacle_guest.json"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"
BASE = "https://guest.api.pinnacle.com/v1"
TENNIS_SPORT_ID = 33
CACHE_TTL_S = 300


def _get(url: str) -> object:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": UA})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_cache() -> dict:
    if not CACHE.exists():
        return {"events": [], "fetched_at": None}
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {"events": [], "fetched_at": None}


def _save_cache(events: list[dict]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps(
            {"fetched_at": datetime.now(timezone.utc).isoformat(), "events": events},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_participants(participants: list) -> tuple[str, str]:
    names = [str(p.get("name") or "").strip() for p in participants if p.get("name")]
    if len(names) >= 2:
        return names[0], names[1]
    return "", ""


def fetch_tennis_fixtures(*, force: bool = False) -> list[dict]:
    """Scarica matchups tennis da guest API Pinnacle."""
    cached = _load_cache()
    ts = cached.get("fetched_at")
    if not force and ts:
        try:
            fetched = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - fetched).total_seconds()
            if age < CACHE_TTL_S and cached.get("events"):
                return cached["events"]
        except Exception:
            pass

    events: list[dict] = []
    try:
        leagues = _get(f"{BASE}/sports/{TENNIS_SPORT_ID}/leagues") or []
        for league in leagues[:40]:
            lid = league.get("id")
            if lid is None:
                continue
            try:
                matchups = _get(f"{BASE}/leagues/{lid}/matchups") or []
            except (HTTPError, URLError, RuntimeError):
                continue
            for m in matchups:
                mid = m.get("id")
                parts = m.get("participants") or []
                pa, pb = _parse_participants(parts)
                if not pa or not pb:
                    continue
                start = m.get("startTime") or m.get("starts")
                events.append(
                    {
                        "event_id": mid,
                        "league_id": lid,
                        "league": league.get("name"),
                        "player_a": pa,
                        "player_b": pb,
                        "starts": start,
                        "odds_a": None,
                        "odds_b": None,
                    }
                )
            time.sleep(0.05)
    except (HTTPError, URLError, RuntimeError) as exc:
        return cached.get("events") or []

    # Quote straight per event (batch leggero)
    for ev in events[:120]:
        eid = ev.get("event_id")
        if eid is None:
            continue
        try:
            markets = _get(f"{BASE}/guest/markets/straight?eventIds={eid}") or []
        except (HTTPError, URLError, RuntimeError):
            continue
        for mkt in markets:
            if str(mkt.get("type") or "").lower() not in ("moneyline", "match"):
                continue
            prices = mkt.get("prices") or []
            if len(prices) >= 2:
                try:
                    ev["odds_a"] = float(prices[0].get("price") or 0) or None
                    ev["odds_b"] = float(prices[1].get("price") or 0) or None
                except (TypeError, ValueError):
                    pass
                break
        time.sleep(0.04)

    _save_cache(events)
    return events


def lookup_pinnacle_guest(
    player_a: str,
    player_b: str,
    *,
    date: str | None = None,
    events: list[dict] | None = None,
) -> dict | None:
    """Cerca quote Pinnacle guest per match (best-effort name match)."""
    if events is None:
        events = fetch_tennis_fixtures()
    day = str(date or "")[:10]

    for ev in events:
        pa, pb = str(ev.get("player_a") or ""), str(ev.get("player_b") or "")
        direct = _last_name(player_a) == _last_name(pa) and _last_name(player_b) == _last_name(pb)
        swap = _last_name(player_a) == _last_name(pb) and _last_name(player_b) == _last_name(pa)
        if not (direct or swap):
            continue
        if day and ev.get("starts"):
            try:
                ed = str(ev["starts"])[:10]
                if abs((datetime.fromisoformat(ed) - datetime.fromisoformat(day)).days) > 2:
                    continue
            except ValueError:
                pass
        oa, ob = ev.get("odds_a"), ev.get("odds_b")
        if not oa or not ob or float(oa) <= 1.01 or float(ob) <= 1.01:
            continue
        if swap:
            oa, ob = ob, oa
        return {"a": float(oa), "b": float(ob), "source": "pinnacle_guest", "event_id": ev.get("event_id")}
    return None
