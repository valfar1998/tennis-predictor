"""Quote Exchange Betfair per tennis ATP/WTA (Delayed App Key).

Login: identitysso.betfair.it
Betting: api.betfair.com Exchange JSON-RPC.

Credenziali:
  .env  BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD
  oppure data/raw/betfair.appkey (solo Delayed Key)
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ENV_PATH = ROOT / ".env"
KEY_PATH = RAW / "betfair.appkey"
SESSION_PATH = RAW / "betfair.session.json"
CACHE = RAW / "betfair_odds.json"

LOGIN_URL = "https://identitysso.betfair.it/api/login"
KEEPALIVE_URL = "https://identitysso.betfair.it/api/keepAlive"
BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0; +local)"

TENNIS_EVENT_TYPE = "2"
MARKET_TYPES = ("MATCH_ODDS",)
SESSION_MAX_AGE_H = 6.0
BOOK_BATCH = 20


def _parse_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _load_env() -> dict[str, str]:
    data = _parse_env(ENV_PATH)
    for key, val in data.items():
        if key.startswith("BETFAIR_") and val and key not in os.environ:
            os.environ[key] = val
    return data


def _app_key() -> str | None:
    _load_env()
    val = (os.environ.get("BETFAIR_APP_KEY") or "").strip()
    if val:
        return val
    if KEY_PATH.exists():
        val = KEY_PATH.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def _credentials() -> tuple[str | None, str | None]:
    _load_env()
    user = (os.environ.get("BETFAIR_USERNAME") or "").strip()
    pwd = (os.environ.get("BETFAIR_PASSWORD") or "").strip()
    return (user or None, pwd or None)


def login_configured() -> bool:
    user, pwd = _credentials()
    return bool(_app_key() and user and pwd)


def _post_form(url: str, data: dict[str, str], headers: dict[str, str]) -> dict:
    req = Request(
        url,
        data=urlencode(data).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json", **headers},
    )
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _rpc(method: str, params: dict, token: str, app_key: str) -> object:
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    req = Request(
        BETTING_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Application": app_key,
            "X-Authentication": token,
            "User-Agent": UA,
        },
    )
    with urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if isinstance(body, list):
        body = body[0] if body else {}
    if body.get("error"):
        err = body["error"]
        data = err.get("data") if isinstance(err, dict) else err
        raise RuntimeError(f"{method} fallito: {data or err}")
    return body.get("result")


def _read_session() -> str | None:
    if not SESSION_PATH.exists():
        return None
    try:
        data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    token = str(data.get("token") or "").strip()
    ts = str(data.get("saved_at") or "")
    if not token or not ts:
        return None
    try:
        saved = datetime.fromisoformat(ts)
        if saved.tzinfo is None:
            saved = saved.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - saved).total_seconds() / 3600
    except ValueError:
        return None
    if age_h > SESSION_MAX_AGE_H:
        return None
    return token


def _write_session(token: str) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(
        json.dumps({"token": token, "saved_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def login(*, force: bool = False) -> str:
    app_key = _app_key()
    if not app_key:
        raise RuntimeError("BETFAIR_APP_KEY assente")
    if not force:
        cached = _read_session()
        if cached:
            try:
                alive = _post_form(
                    KEEPALIVE_URL,
                    {},
                    {"X-Application": app_key, "X-Authentication": cached},
                )
                if str(alive.get("status") or "").upper() == "SUCCESS":
                    return cached
            except Exception:
                pass
    user, pwd = _credentials()
    if not user or not pwd:
        raise RuntimeError("Manca BETFAIR_USERNAME o BETFAIR_PASSWORD nel .env")
    result = _post_form(
        LOGIN_URL,
        {"username": user, "password": pwd},
        {"X-Application": app_key},
    )
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Login Betfair fallito: {result.get('error') or result.get('errorCode') or result}")
    token = str(result.get("token") or "").strip()
    if not token:
        raise RuntimeError(f"Login Betfair senza token: {result}")
    _write_session(token)
    return token


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split_event(name: str) -> tuple[str, str]:
    for sep in (" v ", " vs ", " - ", " @ "):
        if sep in name:
            a, b = name.split(sep, 1)
            return a.strip(), b.strip()
    return name.strip(), ""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip()


def _last_name(name: str) -> str:
    parts = _norm(name).replace(".", "").split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts[-1]) <= 2:
        return parts[0]
    return parts[-1]


def _player_match(a: str, b: str) -> bool:
    from modules.data_update.entity_resolution import player_side_match

    return player_side_match(a, b)


def _best_back(runner: dict) -> float | None:
    ex = runner.get("ex") or {}
    backs = ex.get("availableToBack") or []
    if backs:
        try:
            return round(float(backs[0]["price"]), 3)
        except (TypeError, ValueError, KeyError):
            pass
    last = runner.get("lastPriceTraded")
    if last is None:
        return None
    try:
        return round(float(last), 3)
    except (TypeError, ValueError):
        return None


def _catalogue_window(token: str, app_key: str, start: datetime, end: datetime) -> list[dict]:
    return list(
        _rpc(
            "listMarketCatalogue",
            {
                "filter": {
                    "eventTypeIds": [TENNIS_EVENT_TYPE],
                    "marketTypeCodes": list(MARKET_TYPES),
                    "marketStartTime": {"from": _iso(start), "to": _iso(end)},
                },
                "maxResults": 1000,
                "marketProjection": [
                    "COMPETITION",
                    "EVENT",
                    "MARKET_START_TIME",
                    "MARKET_DESCRIPTION",
                    "RUNNER_DESCRIPTION",
                ],
            },
            token,
            app_key,
        )
        or []
    )


def _catalogue_range(token: str, app_key: str, start: datetime, end: datetime, *, depth: int = 0) -> list[dict]:
    try:
        rows = _catalogue_window(token, app_key, start, end)
        time.sleep(0.08)
        return rows
    except RuntimeError as exc:
        if "TOO_MUCH_DATA" not in str(exc) or depth >= 5:
            raise
        mid = start + (end - start) / 2
        left = _catalogue_range(token, app_key, start, mid, depth=depth + 1)
        right = _catalogue_range(token, app_key, mid, end, depth=depth + 1)
        return left + right


def _market_books_chunk(
    token: str,
    app_key: str,
    market_ids: list[str],
    *,
    depth: int = 0,
) -> list[dict]:
    if not market_ids:
        return []
    try:
        books = _rpc(
            "listMarketBook",
            {
                "marketIds": market_ids,
                "priceProjection": {
                    "priceData": ["EX_BEST_OFFERS", "EX_TRADED", "SP_AVAILABLE", "SP_TRADED"],
                    "virtualise": True,
                },
            },
            token,
            app_key,
        ) or []
        time.sleep(0.12)
        return books
    except RuntimeError as exc:
        if "TOO_MUCH_DATA" not in str(exc) or len(market_ids) <= 1 or depth >= 8:
            raise
        mid = len(market_ids) // 2
        left = _market_books_chunk(token, app_key, market_ids[:mid], depth=depth + 1)
        right = _market_books_chunk(token, app_key, market_ids[mid:], depth=depth + 1)
        return left + right


def _market_books(token: str, app_key: str, market_ids: list[str]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(market_ids), BOOK_BATCH):
        chunk = market_ids[i : i + BOOK_BATCH]
        out.extend(_market_books_chunk(token, app_key, chunk))
    return out


def _group_events(catalogue: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for mkt in catalogue:
        event = mkt.get("event") or {}
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        name = str(event.get("name") or "")
        player_a, player_b = _split_event(name)
        row = grouped.setdefault(
            event_id,
            {
                "event_id": event_id,
                "event_name": name,
                "player_a": player_a,
                "player_b": player_b,
                "commence_time": event.get("openDate") or mkt.get("marketStartTime"),
                "competition": (mkt.get("competition") or {}).get("name") or "",
                "markets": {},
                "runners": {},
            },
        )
        mid = str(mkt.get("marketId") or "")
        runners = {
            int(r["selectionId"]): str(r.get("runnerName") or "")
            for r in (mkt.get("runners") or [])
            if r.get("selectionId") is not None
        }
        mtype = str((mkt.get("description") or {}).get("marketType") or "")
        mname = str(mkt.get("marketName") or "").lower()
        if mtype == "MATCH_ODDS" or "match odds" in mname:
            row["markets"]["MATCH_ODDS"] = mid
            row["runners"]["MATCH_ODDS"] = runners
    return grouped


def _fetch_catalogue_and_books(token: str, app_key: str, *, days: int) -> list[dict]:
    now = datetime.now(timezone.utc)
    catalogue: list[dict] = []
    steps = max(1, int(days) * 4)
    from modules.ops_progress import log_item

    for i in range(steps):
        start = now + timedelta(hours=6 * i)
        end = now + timedelta(hours=6 * (i + 1))
        catalogue.extend(_catalogue_range(token, app_key, start, end))
        log_item(i + 1, steps, "catalogo Betfair")
    grouped = _group_events(catalogue)
    market_ids = [mid for row in grouped.values() for mid in row["markets"].values() if mid]
    books = {str(b.get("marketId")): b for b in _market_books(token, app_key, market_ids)}

    events: list[dict] = []
    for row in grouped.values():
        if not row["player_a"] or not row["player_b"]:
            continue
        ev = {
            "event_id": row["event_id"],
            "event_name": row["event_name"],
            "player_a": row["player_a"],
            "player_b": row["player_b"],
            "commence_time": row["commence_time"],
            "competition": row["competition"],
            "odd_a": None,
            "odd_b": None,
            "runner_a": None,
            "runner_b": None,
        }
        match_id = row["markets"].get("MATCH_ODDS")
        if match_id and match_id in books:
            names = row["runners"].get("MATCH_ODDS") or {}
            unmatched: list[tuple[str, float]] = []
            for runner in books[match_id].get("runners") or []:
                sid = runner.get("selectionId")
                name = names.get(int(sid) if sid is not None else -1, "")
                price = _runner_price(runner)
                if price is None:
                    continue
                if _player_match(row["player_a"], name):
                    ev["odd_a"] = price
                    ev["runner_a"] = name
                elif _player_match(row["player_b"], name):
                    ev["odd_b"] = price
                    ev["runner_b"] = name
                elif name:
                    unmatched.append((name, price))
            if ev["odd_a"] is None or ev["odd_b"] is None:
                for name, price in unmatched:
                    if ev["odd_a"] is None and _player_match(row["player_a"], name):
                        ev["odd_a"] = price
                        ev["runner_a"] = name
                    elif ev["odd_b"] is None and _player_match(row["player_b"], name):
                        ev["odd_b"] = price
                        ev["runner_b"] = name
            if ev["odd_a"] and ev["odd_b"]:
                from modules.data_update.entity_resolution import align_odds_to_players

                aligned = align_odds_to_players(
                    row["player_a"],
                    row["player_b"],
                    ev["odd_a"],
                    ev["odd_b"],
                    runner_a=ev.get("runner_a"),
                    runner_b=ev.get("runner_b"),
                )
                if aligned.get("blocked"):
                    continue
                ev["odd_a"] = aligned["odd_a"]
                ev["odd_b"] = aligned["odd_b"]
                if aligned.get("swapped"):
                    ev["odds_swapped"] = True
                ev["odds_verified"] = aligned.get("verified", False)
                events.append(ev)
    return events


def fetch_betfair_odds(*, force: bool = False, days: int = 7, max_age_hours: float = 1.0) -> dict:
    """Scarica quote MATCH_ODDS Exchange per tennis in arrivo."""
    if not force and CACHE.exists():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            ts = str(data.get("fetched_at") or "")
            if ts:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                age_s = (datetime.now(timezone.utc) - fetched).total_seconds()
                if age_s < max_age_hours * 3600:
                    events = data.get("events") or []
                    return {
                        "ok": True,
                        "n_events": len(events),
                        "from_cache": True,
                        "events": events,
                    }
        except Exception:
            pass

    app_key = _app_key()
    if not app_key:
        return {"ok": False, "error": "BETFAIR_APP_KEY non trovata", "n_events": 0, "events": [], "from_cache": False}

    try:
        token = login(force=force)
        try:
            events = _fetch_catalogue_and_books(token, app_key, days=days)
        except RuntimeError as exc:
            if "INVALID_SESSION" not in str(exc):
                raise
            token = login(force=True)
            events = _fetch_catalogue_and_books(token, app_key, days=days)

        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_events": len(events),
            "events": events,
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ok Betfair tennis: {len(events)} eventi", flush=True)
        return {"ok": True, "n_events": len(events), "from_cache": False, "events": events}
    except HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}", "n_events": 0, "events": [], "from_cache": False}
    except (URLError, TimeoutError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc), "n_events": 0, "events": [], "from_cache": False}


def load_betfair_cache() -> list[dict]:
    if not CACHE.exists():
        return []
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        return data.get("events") or []
    except Exception:
        return []


def _last_traded(runner: dict) -> float | None:
    last = runner.get("lastPriceTraded")
    if last is None:
        return None
    try:
        return round(float(last), 3)
    except (TypeError, ValueError):
        return None


def _runner_close_price(runner: dict) -> float | None:
    """BSP (actual SP) oppure LTP — chiusura Exchange."""
    sp = runner.get("sp") or {}
    for key in ("actualSP", "nearPrice", "farPrice"):
        val = sp.get(key)
        if val is not None:
            try:
                f = float(val)
                if f > 1.01:
                    return round(f, 3)
            except (TypeError, ValueError):
                pass
    return _last_traded(runner) or _best_back(runner)


def lookup_betfair_close(
    player_a: str,
    player_b: str,
    *,
    event_id: str | None = None,
    match_date: str | None = None,
    bet_odds: dict | None = None,
    events: list[dict] | None = None,
) -> dict | None:
    """Quote Betfair al kickoff (LTP/back) come proxy chiusura Pinnacle de-vigged."""
    ev = None
    if events is None:
        events = load_betfair_cache()
    if event_id:
        for row in events:
            if str(row.get("event_id")) == str(event_id):
                ev = row
                break
    if ev is None:
        ev = lookup_betfair_match(player_a, player_b, events=events, match_date=match_date)
    if not ev:
        if bet_odds and bet_odds.get("a") and bet_odds.get("b"):
            return {
                "a": float(bet_odds["a"]),
                "b": float(bet_odds["b"]),
                "source": "betfair_bet_snapshot",
            }
        return None
    oa, ob = ev.get("odd_a"), ev.get("odd_b")
    if not oa or not ob:
        return None
    return {
        "a": float(oa),
        "b": float(ob),
        "source": "betfair_ltp",
        "event_id": ev.get("event_id"),
    }


def lookup_betfair_match(
    player_a: str,
    player_b: str,
    *,
    events: list[dict] | None = None,
    match_date: str | None = None,
) -> dict | None:
    """Cerca un match tennis nella cache Betfair per nome giocatore."""
    if events is None:
        events = load_betfair_cache()
    if not events:
        return None
    md = None
    if match_date:
        try:
            md = date.fromisoformat(str(match_date)[:10])
        except ValueError:
            pass
    for ev in events:
        pa, pb = str(ev.get("player_a") or ""), str(ev.get("player_b") or "")
        direct = _player_match(player_a, pa) and _player_match(player_b, pb)
        swap = _player_match(player_a, pb) and _player_match(player_b, pa)
        if not (direct or swap):
            continue
        if md:
            ct = str(ev.get("commence_time") or "")[:10]
            try:
                if abs((date.fromisoformat(ct) - md).days) > 1:
                    continue
            except ValueError:
                pass
        if swap:
            from modules.data_update.entity_resolution import align_odds_to_players

            aligned = align_odds_to_players(
                player_a,
                player_b,
                ev.get("odd_b"),
                ev.get("odd_a"),
                runner_a=ev.get("runner_b"),
                runner_b=ev.get("runner_a"),
            )
            return {
                **ev,
                "player_a": player_a,
                "player_b": player_b,
                "odd_a": aligned["odd_a"],
                "odd_b": aligned["odd_b"],
                "odds_swapped": True,
            }
        from modules.data_update.entity_resolution import align_odds_to_players

        aligned = align_odds_to_players(
            player_a,
            player_b,
            ev.get("odd_a"),
            ev.get("odd_b"),
            runner_a=ev.get("runner_a"),
            runner_b=ev.get("runner_b"),
        )
        if not aligned.get("blocked"):
            return {**ev, "odd_a": aligned["odd_a"], "odd_b": aligned["odd_b"]}
        return ev
    return None


SETTLED_CACHE = RAW / "betfair_settled.json"


def fetch_betfair_settled_results(*, days: int = 14, force: bool = False, max_age_hours: float = 2.0) -> dict:
    """Mercati MATCH_ODDS chiusi con runner WINNER (ultimi N giorni)."""
    if not force and SETTLED_CACHE.exists():
        try:
            data = json.loads(SETTLED_CACHE.read_text(encoding="utf-8"))
            ts = str(data.get("fetched_at") or "")
            if ts:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                age_s = (datetime.now(timezone.utc) - fetched).total_seconds()
                if age_s < max_age_hours * 3600 and data.get("results") is not None:
                    return data
        except Exception:
            pass

    app_key = _app_key()
    if not app_key:
        return {"ok": False, "error": "BETFAIR_APP_KEY non trovata", "results": []}

    try:
        token = login(force=False)
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)
        from modules.ops_progress import log_step

        log_step(1, 3, "Betfair settled: catalogo mercati...")
        catalogue = _catalogue_range(token, app_key, start, now)
        grouped = _group_events(catalogue)
        market_ids = [mid for row in grouped.values() for mid in row["markets"].values() if mid]
        log_step(2, 3, f"Betfair settled: quote {len(market_ids)} mercati...")
        books = {str(b.get("marketId")): b for b in _market_books(token, app_key, market_ids)}

        results: list[dict] = []
        for row in grouped.values():
            mid = row["markets"].get("MATCH_ODDS")
            if not mid or str(mid) not in books:
                continue
            book = books[str(mid)]
            if str(book.get("status") or "").upper() not in ("CLOSED", "SETTLED"):
                continue
            names = row["runners"].get("MATCH_ODDS") or {}
            pa, pb = row["player_a"], row["player_b"]
            winner_name = loser_name = None
            odd_a = odd_b = None
            for runner in book.get("runners") or []:
                sid = runner.get("selectionId")
                name = names.get(int(sid) if sid is not None else -1, "")
                status = str(runner.get("status") or "").upper()
                price = _runner_close_price(runner)
                if name and (_player_match(name, pa) or _player_match(pa, name)):
                    odd_a = price
                elif name and (_player_match(name, pb) or _player_match(pb, name)):
                    odd_b = price
                if status == "WINNER":
                    winner_name = name
                elif status == "LOSER" and name:
                    loser_name = name
            if not winner_name:
                continue
            if _player_match(winner_name, pa):
                winner, loser = pa, pb if loser_name is None else loser_name
            elif _player_match(winner_name, pb):
                winner, loser = pb, pa if loser_name is None else loser_name
            else:
                winner, loser = winner_name, loser_name or (pb if winner_name != pa else pa)
            results.append(
                {
                    "event_id": row["event_id"],
                    "player_a": pa,
                    "player_b": pb,
                    "winner": winner,
                    "loser": loser,
                    "odd_a": odd_a,
                    "odd_b": odd_b,
                    "commence_time": row.get("commence_time"),
                    "date": str(row.get("commence_time") or "")[:10],
                    "competition": row.get("competition"),
                    "source": "betfair_settled",
                }
            )

        log_step(3, 3, f"Betfair settled: {len(results)} risultati")
        payload = {
            "ok": True,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_results": len(results),
            "results": results,
        }
        SETTLED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SETTLED_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    except Exception as exc:
        return {"ok": False, "error": str(exc), "results": []}


def load_betfair_settled_results(*, days: int = 14) -> list[dict]:
    if SETTLED_CACHE.exists():
        try:
            data = json.loads(SETTLED_CACHE.read_text(encoding="utf-8"))
            if data.get("results"):
                return data["results"]
        except Exception:
            pass
    info = fetch_betfair_settled_results(days=days, force=False)
    return info.get("results") or []


def lookup_betfair_settled_close(
    player_a: str,
    player_b: str,
    *,
    match_date: str | None = None,
) -> dict | None:
    """Chiusura da mercati Betfair già CLOSED/SETTLED (BSP o LTP)."""
    rows = []
    if SETTLED_CACHE.exists():
        try:
            rows = json.loads(SETTLED_CACHE.read_text(encoding="utf-8")).get("results") or []
        except Exception:
            rows = []
    if not rows:
        return None
    md = None
    if match_date:
        try:
            md = date.fromisoformat(str(match_date)[:10])
        except ValueError:
            pass
    for ev in rows:
        pa, pb = str(ev.get("player_a") or ""), str(ev.get("player_b") or "")
        direct = _player_match(player_a, pa) and _player_match(player_b, pb)
        swap = _player_match(player_a, pb) and _player_match(player_b, pa)
        if not (direct or swap):
            continue
        if md:
            day = str(ev.get("date") or ev.get("commence_time") or "")[:10]
            try:
                if abs((date.fromisoformat(day) - md).days) > 1:
                    continue
            except ValueError:
                pass
        oa, ob = ev.get("odd_a"), ev.get("odd_b")
        if swap:
            oa, ob = ob, oa
        if not oa or not ob:
            continue
        return {
            "a": float(oa),
            "b": float(ob),
            "source": "betfair_settled",
            "event_id": ev.get("event_id"),
        }
    return None
