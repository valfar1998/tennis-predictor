"""Risultati tennis da FlashScore / diretta.it (feed ninja, best-effort)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "flashscore_results.json"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"
FEED_BASE = "https://global.flashscore.ninja/2/x/feed"
FEED_ALT = "https://d.flashscore.com/x/feed"
HTML_SOURCES = (
    "https://www.flashscore.com/tennis/",
    "https://www.diretta.it/tennis/",
    "https://www.flashscore.it/tennis/",
)
# f_{sport}_{day_offset}_{tab}_{lang}_1 — sport 2 = tennis, tab 3 = results
FEED_URLS = (
    f"{FEED_BASE}/f_2_0_3_en_1",
    f"{FEED_BASE}/f_2_-1_3_en_1",
    f"{FEED_ALT}/f_2_0_3_en_1",
    f"{FEED_ALT}/f_2_-1_3_en_1",
)


def _fetch_feed(url: str) -> str | None:
    try:
        req = Request(
            url,
            headers={
                "User-Agent": UA,
                "Accept": "*/*",
                "Referer": "https://www.flashscore.com/tennis/",
                "Origin": "https://www.flashscore.com",
            },
        )
        with urlopen(req, timeout=25) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _parse_feed(raw: str) -> list[dict]:
    """Parsa feed FlashScore (record separati da ~, campi KEY÷value)."""
    events: dict[str, dict[str, str]] = {}
    current: str | None = None
    for part in raw.split("~"):
        if not part or "÷" not in part:
            continue
        key, _, val = part.partition("÷")
        if key == "AA":
            current = val
            events.setdefault(current, {})
        elif current:
            events[current][key] = val

    rows: list[dict] = []
    for eid, ev in events.items():
        p1 = (ev.get("CX") or ev.get("AE") or "").strip()
        p2 = (ev.get("CY") or ev.get("AF") or "").strip()
        if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
            continue
        status = str(ev.get("AC") or ev.get("AB") or "")
        # 3 = finished, 4 = walkover/retired variants on many feeds
        if status and status not in ("3", "4", "5"):
            continue
        score = str(ev.get("AG") or ev.get("AT") or ev.get("AU") or "").strip()
        ts = ev.get("AD")
        match_date = None
        if ts and str(ts).isdigit():
            try:
                match_date = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
            except (OSError, ValueError, OverflowError):
                match_date = None

        winner = _winner_from_feed(ev, p1, p2, score)
        if not winner:
            continue
        loser = p2 if winner == p1 else p1
        rows.append(
            {
                "player_a": p1,
                "player_b": p2,
                "winner": winner,
                "loser": loser,
                "score": score or None,
                "date": match_date,
                "event_id": eid,
                "source": "flashscore",
            }
        )
    return rows


def _winner_from_feed(ev: dict[str, str], p1: str, p2: str, score: str) -> str | None:
    """Determina vincitore da flag feed o punteggio set."""
    side = str(ev.get("AS") or ev.get("AZ") or ev.get("NI") or "").strip()
    if side == "1":
        return p1
    if side == "2":
        return p2
    if score:
        sets = re.findall(r"(\d+)", score.replace(",", " "))
        if len(sets) >= 2:
            try:
                s1, s2 = int(sets[0]), int(sets[1])
                if s1 > s2:
                    return p1
                if s2 > s1:
                    return p2
            except ValueError:
                pass
    return None


def _parse_html_results(html: str) -> list[dict]:
    """Fallback: estrae coppie home/away da HTML FlashScore/diretta (partite finite)."""
    rows: list[dict] = []
    # Blocchi match statici (finite)
    for block in re.findall(
        r'event__match[^"]*event__match--static[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
        html,
        flags=re.I | re.S,
    ):
        home_m = re.search(
            r'event__participant[^"]*--home[^"]*"[^>]*>([^<]+)<',
            block,
            flags=re.I,
        )
        away_m = re.search(
            r'event__participant[^"]*--away[^"]*"[^>]*>([^<]+)<',
            block,
            flags=re.I,
        )
        if not home_m or not away_m:
            continue
        p1, p2 = home_m.group(1).strip(), away_m.group(1).strip()
        score_m = re.search(r'event__score[^>]*>([^<]+)<', block, flags=re.I)
        score = score_m.group(1).strip() if score_m else None
        winner = _winner_from_feed({"AS": "1"}, p1, p2, score or "")
        if not winner and score:
            winner = _winner_from_feed({}, p1, p2, score)
        if not winner:
            continue
        rows.append(
            {
                "player_a": p1,
                "player_b": p2,
                "winner": winner,
                "loser": p2 if winner == p1 else p1,
                "score": score,
                "date": None,
                "source": "flashscore_html",
            }
        )
    return rows


def _fetch_html_results() -> list[dict]:
    rows: list[dict] = []
    for url in HTML_SOURCES:
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
            with urlopen(req, timeout=25) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            rows.extend(_parse_html_results(html))
        except Exception:
            continue
    return rows


def fetch_flashscore_results(*, force: bool = False, max_age_hours: float = 0.5) -> dict:
    """Scarica risultati tennis recenti (oggi + 2 giorni)."""
    if not force and is_fresh(CACHE, max_age_hours=max_age_hours) and CACHE.is_file():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if cached.get("matches"):
                return cached
        except Exception:
            pass

    matches: list[dict] = []
    seen: set[str] = set()
    errors: list[str] = []
    for url in FEED_URLS:
        raw = _fetch_feed(url)
        if not raw:
            errors.append(f"fetch_fail:{url}")
            continue
        for row in _parse_feed(raw):
            key = "|".join(sorted([row["player_a"].lower(), row["player_b"].lower(), str(row.get("date") or "")]))
            if key in seen:
                continue
            seen.add(key)
            matches.append(row)

    if not matches:
        for row in _fetch_html_results():
            key = "|".join(sorted([row["player_a"].lower(), row["player_b"].lower()]))
            if key in seen:
                continue
            seen.add(key)
            matches.append(row)

    info = {
        "ok": bool(matches),
        "source": "flashscore",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "matches": matches,
        "errors": errors or None,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load_flashscore_results() -> list[dict]:
    if CACHE.is_file():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            return data.get("matches") or []
        except Exception:
            pass
    info = fetch_flashscore_results(force=False)
    return info.get("matches") or []
