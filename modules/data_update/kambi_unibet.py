"""Palinsesto tennis Unibet via Kambi Guest API (ATP/WTA/Challenger/ITF)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "kambi_unibet_odds.json"
DEFAULT_CLIENT = "ub"
DEFAULT_LANG = "it_IT"
DEFAULT_MARKET = "IT"
DEFAULT_MAX_AGE_MIN = 45.0
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
URL_TEMPLATES = (
    "https://eu-offering-api.kambi.com/v2018/{client}/listView/tennis.json",
    "https://eu-offering.kambicdn.org/offering/v2018/{client}/listView/tennis.json",
)
SKIP_STATES = frozenset({"FINISHED", "CANCELLED", "ABANDONED", "POSTPONED", "SUSPENDED"})
MATCH_CRITERIA = (
    "match",
    "match odds",
    "vincitore",
    "winner",
    "moneyline",
    "1x2",
)


def _enabled() -> bool:
    return (os.environ.get("KAMBI_UNIBET_ENABLED") or "1").strip().lower() in ("1", "true", "yes")


def _client() -> str:
    return (os.environ.get("KAMBI_CLIENT") or DEFAULT_CLIENT).strip() or DEFAULT_CLIENT


def _params() -> dict[str, str]:
    return {
        "lang": (os.environ.get("KAMBI_LANG") or DEFAULT_LANG).strip() or DEFAULT_LANG,
        "market": (os.environ.get("KAMBI_MARKET") or DEFAULT_MARKET).strip() or DEFAULT_MARKET,
        "useCombined": "true",
    }


def _headers(host: str) -> dict[str, str]:
    referer = (os.environ.get("KAMBI_REFERER") or "https://www.unibet.it/").strip()
    return {
        "User-Agent": UA,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
        "Referer": referer,
        "Origin": referer.rstrip("/"),
        "Host": host,
    }


def _http_get_json(url: str, *, params: dict[str, str], host: str) -> Any:
    try:
        from curl_cffi import requests as http

        resp = http.get(
            url,
            params=params,
            headers=_headers(host),
            timeout=30,
            impersonate="chrome",
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    try:
        import requests as http

        resp = http.get(url, params=params, headers=_headers(host), timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _fetch_payload() -> tuple[Any, str | None]:
    client = _client()
    params = _params()
    errors: list[str] = []
    for template in URL_TEMPLATES:
        url = template.format(client=client)
        host = url.split("/")[2]
        payload = _http_get_json(url, params=params, host=host)
        if payload is not None:
            return payload, None
        errors.append(host)
        time.sleep(0.5)
    return None, f"Kambi non raggiungibile ({', '.join(errors)})"


def _kambi_decimal(odds_raw: Any) -> float | None:
    if odds_raw is None:
        return None
    try:
        val = float(odds_raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val > 20:
        val = val / 1000.0
    return round(val, 3) if val > 1.01 else None


def _format_player_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    if "," in text:
        last, first = text.split(",", 1)
        return f"{first.strip()} {last.strip()}".strip()
    return text


def _path_label(path: list[dict] | None) -> str:
    if not isinstance(path, list):
        return ""
    parts: list[str] = []
    for node in path:
        if not isinstance(node, dict):
            continue
        label = str(node.get("englishName") or node.get("name") or "").strip()
        if label and label.lower() not in ("tennis",):
            parts.append(label)
    return " / ".join(parts)


def _infer_level(*, group: str, path_label: str) -> str:
    blob = f"{group} {path_label}".lower()
    if any(k in blob for k in ("us open", "wimbledon", "roland garros", "australian open", "grand slam")):
        return "G"
    if any(k in blob for k in ("masters", "1000", "miami", "indian wells")):
        return "M"
    if "challenger" in blob:
        return "C"
    if "itf" in blob:
        return "S"
    if "wta" in blob:
        return "A"
    if "atp" in blob:
        return "A"
    return "A"


def _is_match_offer(offer: dict) -> bool:
    criterion = str((offer.get("criterion") or {}).get("englishLabel") or "").lower()
    offer_type = str((offer.get("betOfferType") or {}).get("englishName") or "").lower()
    blob = f"{criterion} {offer_type}"
    if "set" in blob and "match" not in blob:
        return False
    if any(k in blob for k in ("handicap", "total", "games", "set", "correct score", "tie break")):
        return False
    return any(k in blob for k in MATCH_CRITERIA) or offer_type in ("match", "kamp", "head to head")


def _pick_match_odds(
    offers: list[dict],
    *,
    home_name: str,
    away_name: str,
) -> tuple[float | None, float | None, str | None, str | None]:
    home_fmt = _format_player_name(home_name)
    away_fmt = _format_player_name(away_name)

    for offer in offers:
        if not isinstance(offer, dict) or offer.get("closed"):
            continue
        if not _is_match_offer(offer):
            continue
        outcomes = [o for o in (offer.get("outcomes") or []) if isinstance(o, dict)]
        open_outcomes = [o for o in outcomes if str(o.get("status") or "OPEN").upper() == "OPEN"]
        if len(open_outcomes) != 2:
            continue

        by_type: dict[str, dict] = {}
        by_participant: dict[str, dict] = {}
        for outcome in open_outcomes:
            otype = str(outcome.get("type") or "").upper()
            participant = _format_player_name(
                str(outcome.get("participant") or outcome.get("englishLabel") or outcome.get("label") or "")
            )
            if otype:
                by_type[otype] = outcome
            if participant:
                by_participant[participant.lower()] = outcome

        odd_home = odd_away = None
        runner_home = runner_away = None

        if "OT_ONE" in by_type and "OT_TWO" in by_type:
            odd_home = _kambi_decimal(by_type["OT_ONE"].get("odds"))
            odd_away = _kambi_decimal(by_type["OT_TWO"].get("odds"))
            runner_home = home_fmt
            runner_away = away_fmt
        else:
            for participant, outcome in by_participant.items():
                odds = _kambi_decimal(outcome.get("odds"))
                if odds is None:
                    continue
                if participant and participant in home_fmt.lower():
                    odd_home = odds
                    runner_home = home_fmt
                elif participant and participant in away_fmt.lower():
                    odd_away = odds
                    runner_away = away_fmt
            if odd_home is None or odd_away is None:
                ordered = []
                for outcome in open_outcomes:
                    odds = _kambi_decimal(outcome.get("odds"))
                    if odds is None:
                        continue
                    label = _format_player_name(
                        str(outcome.get("participant") or outcome.get("englishLabel") or outcome.get("label") or "")
                    )
                    ordered.append((label, odds))
                if len(ordered) == 2:
                    odd_home, runner_home = ordered[0][1], ordered[0][0] or home_fmt
                    odd_away, runner_away = ordered[1][1], ordered[1][0] or away_fmt

        if odd_home and odd_away:
            return odd_home, odd_away, runner_home, runner_away
    return None, None, None, None


def _normalize_events(payload: Any) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(payload, dict):
        return rows

    for item in payload.get("events") or []:
        if not isinstance(item, dict):
            continue
        event = item.get("event") or {}
        if not isinstance(event, dict):
            continue
        state = str(event.get("state") or "").upper()
        if state in SKIP_STATES:
            continue

        home = _format_player_name(str(event.get("homeName") or ""))
        away = _format_player_name(str(event.get("awayName") or ""))
        if not home or not away:
            name = str(event.get("name") or event.get("englishName") or "")
            if " - " in name:
                left, right = name.split(" - ", 1)
                home = home or _format_player_name(left)
                away = away or _format_player_name(right)
        if not home or not away:
            continue

        odd_a, odd_b, runner_a, runner_b = _pick_match_odds(
            item.get("betOffers") or [],
            home_name=home,
            away_name=away,
        )
        if not odd_a or not odd_b:
            continue

        group = str(event.get("group") or event.get("englishGroup") or "")
        path_label = _path_label(event.get("path"))
        competition = path_label or group or "Tennis"
        start = str(event.get("start") or "")
        rows.append(
            {
                "event_id": f"kambi:{event.get('id')}",
                "event_name": str(event.get("englishName") or event.get("name") or f"{home} vs {away}"),
                "player_a": home,
                "player_b": away,
                "commence_time": start,
                "competition": competition,
                "tourney_level": _infer_level(group=group, path_label=path_label),
                "odd_a": odd_a,
                "odd_b": odd_b,
                "runner_a": runner_a or home,
                "runner_b": runner_b or away,
                "odds_source": "kambi_unibet",
                "state": state or None,
                "tags": list(event.get("tags") or []),
                "is_itf": "itf" in competition.lower(),
                "is_challenger": "challenger" in competition.lower(),
            }
        )
    return rows


def fetch_kambi_tennis_odds(
    *,
    force: bool = False,
    max_age_minutes: float = DEFAULT_MAX_AGE_MIN,
) -> dict[str, Any]:
    """Scarica palinsesto tennis Unibet/Kambi (Guest API, senza chiavi)."""
    if not _enabled():
        return {
            "ok": False,
            "error": "KAMBI_UNIBET_ENABLED=0",
            "n_events": 0,
            "events": [],
        }

    max_age_hours = max(0.05, float(max_age_minutes) / 60.0)
    if not force and is_fresh(CACHE, max_age_hours=max_age_hours) and CACHE.is_file():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if cached.get("events") is not None:
                cached["from_cache"] = True
                return cached
        except Exception:
            pass

    payload, err = _fetch_payload()
    if payload is None:
        if CACHE.is_file():
            try:
                cached = json.loads(CACHE.read_text(encoding="utf-8"))
                cached["ok"] = bool(cached.get("events"))
                cached["error"] = err
                cached["from_cache"] = True
                cached["stale"] = True
                return cached
            except Exception:
                pass
        return {"ok": False, "error": err, "n_events": 0, "events": [], "from_cache": False}

    events = _normalize_events(payload)
    info = {
        "ok": bool(events),
        "source": "kambi_unibet",
        "client": _client(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_events": len(events),
        "n_itf": sum(1 for e in events if e.get("is_itf")),
        "n_challenger": sum(1 for e in events if e.get("is_challenger")),
        "events": events,
        "from_cache": False,
        "error": None if events else "nessun evento tennis con quote match",
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load_kambi_cache() -> list[dict]:
    if not CACHE.is_file():
        return []
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        return data.get("events") or []
    except Exception:
        return []


def merge_odds_events(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Unisce eventi quote: primary (es. Betfair) ha priorità; secondary riempie i buchi."""
    from modules.data_update.entity_resolution import _norm_name

    out = list(primary)
    seen = {
        "|".join(sorted([_norm_name(e.get("player_a", "")), _norm_name(e.get("player_b", ""))]))
        for e in primary
        if e.get("player_a") and e.get("player_b")
    }
    added = 0
    for ev in secondary:
        pa = str(ev.get("player_a") or "")
        pb = str(ev.get("player_b") or "")
        if not pa or not pb:
            continue
        key = "|".join(sorted([_norm_name(pa), _norm_name(pb)]))
        if key in seen:
            continue
        out.append(ev)
        seen.add(key)
        added += 1
    if added:
        print(f"  Kambi/Unibet: +{added} eventi extra (ITF/Challenger/gap-fill)", flush=True)
    return out
