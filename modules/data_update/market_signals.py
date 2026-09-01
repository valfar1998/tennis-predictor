"""Segnali mercato: Moneyway (Arbworld), dropping odds (OddsSafari)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from modules.data_update.entity_resolution import _last_name, _norm_name

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MONEYWAY_CACHE = RAW / "arbworld_moneyway.json"
DROPPING_CACHE = RAW / "oddssafari_dropping.json"

UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"
ARBWORLD_URL = "https://arbworld.net/moneyway/tennis/1x2"
ODDSSAFARI_URL = "https://www.oddssafari.com/dropping-odds/sports/30"


def _get_html(url: str, *, timeout: int = 25) -> str:
    req = Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_odds_cell(text: str) -> dict | None:
    m = re.search(r"([\d.]+)\s+([\d.]+)\s*%\s*[\£€]?\s*([\d,]+)?", text.replace(",", ""))
    if not m:
        return None
    return {
        "odds": float(m.group(1)),
        "volume_pct": float(m.group(2)),
        "volume_gbp": float(m.group(3)) if m.group(3) else None,
    }


def _split_match_label(label: str) -> tuple[str, str, str, str]:
    """Ritorna (date_part, tournament, player_a, player_b)."""
    label = re.sub(r"\s+", " ", label.strip())
    date_m = re.match(r"([A-Za-z]{3}\s+\d{1,2},\s*\d{2}:\d{2})", label)
    date_part = date_m.group(1) if date_m else ""
    rest = label[len(date_part) :].strip() if date_part else label

    sep_pattern = re.compile(r"\s+(?:[—–\-]|vs\.?|v)\s+", re.I)
    m = None
    for match in sep_pattern.finditer(rest):
        m = match
    if not m:
        return date_part, rest, rest, ""

    left, right = rest[: m.start()].strip(), rest[m.end() :].strip()
    tourney, pa = _extract_tourney_and_player(left)
    return date_part, tourney, pa, right


def _extract_tourney_and_player(left: str) -> tuple[str, str]:
    left = left.strip()
    year_player = re.match(r"^(.*\b\d{4})\s+(.+)$", left)
    if year_player:
        return year_player.group(1).strip(), year_player.group(2).strip()
    parts = left.split()
    if len(parts) >= 3:
        return " ".join(parts[:-2]), " ".join(parts[-2:])
    if len(parts) == 2:
        return parts[0], parts[1]
    return left, left


def _players_match(a: str, b: str, x: str, y: str) -> bool:
    la, lb, lx, ly = _last_name(a), _last_name(b), _last_name(x), _last_name(y)
    if not la or not lb or not lx or not ly:
        return False
    direct = la == lx and lb == ly
    swap = la == ly and lb == lx
    return direct or swap


def parse_arbworld_moneyway(html: str) -> list[dict]:
    from bs4 import BeautifulSoup

    rows: list[dict] = []
    soup = BeautifulSoup(html, "lxml")
    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        label = tds[0].get_text(" ", strip=True)
        c1 = _parse_odds_cell(tds[1].get_text(" ", strip=True))
        c2 = _parse_odds_cell(tds[2].get_text(" ", strip=True))
        if not c1 or not c2:
            continue
        date_part, tourney, pa, pb = _split_match_label(label)
        if not pa or not pb:
            continue
        vol_text = tds[3].get_text(" ", strip=True)
        vol_m = re.search(r"[\£€]?\s*([\d,]+)", vol_text.replace(",", ""))
        rows.append({
            "source": "arbworld",
            "date_label": date_part,
            "tourney": tourney,
            "player_a": pa,
            "player_b": pb,
            "odd_a": c1["odds"],
            "odd_b": c2["odds"],
            "volume_pct_a": c1["volume_pct"],
            "volume_pct_b": c2["volume_pct"],
            "total_volume_gbp": float(vol_m.group(1)) if vol_m else None,
        })
    return rows


def parse_oddssafari_dropping(html: str) -> list[dict]:
    from bs4 import BeautifulSoup

    rows: list[dict] = []
    soup = BeautifulSoup(html, "lxml")
    current_league = ""
    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        texts = [td.get_text(" ", strip=True) for td in tds]
        if len(tds) == 2 and texts[0] and not texts[1]:
            current_league = texts[0]
            continue
        if len(tds) < 6:
            continue
        match_label = texts[0]
        body = re.sub(r"^\d{2}/\d{2}\s+\d{2}:\d{2}\s+", "", match_label).strip()
        pp = re.split(r"\s+-\s+", body, maxsplit=1)
        if len(pp) != 2:
            continue
        player_a, player_b = pp[0].strip(), pp[1].strip()
        side = texts[1]
        try:
            open_odds = float(texts[2])
            cur_odds = float(texts[3])
        except ValueError:
            continue
        drop_m = re.search(r"-?\s*([\d.]+)\s*%", texts[4])
        drop_pct = float(drop_m.group(1)) if drop_m else 0.0
        rows.append({
            "source": "oddssafari",
            "league": current_league,
            "datetime_label": match_label.split()[0] + " " + match_label.split()[1] if " " in match_label else "",
            "player_a": player_a,
            "player_b": player_b,
            "side": side,
            "open_odds": open_odds,
            "current_odds": cur_odds,
            "drop_pct": drop_pct,
        })
    return rows


def fetch_moneyway(*, force: bool = False, max_age_minutes: int = 30) -> dict:
    if not force and MONEYWAY_CACHE.is_file():
        try:
            data = json.loads(MONEYWAY_CACHE.read_text(encoding="utf-8"))
            ts = data.get("fetched_at", "")
            if ts:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - fetched).total_seconds() / 60
                if age < max_age_minutes:
                    return {"ok": True, "n": len(data.get("rows", [])), "from_cache": True, "rows": data["rows"]}
        except Exception:
            pass
    try:
        html = _get_html(ARBWORLD_URL)
        rows = parse_arbworld_moneyway(html)
        payload = {"fetched_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
        RAW.mkdir(parents=True, exist_ok=True)
        MONEYWAY_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "n": len(rows), "from_cache": False, "rows": rows}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "n": 0, "rows": [], "from_cache": False}


def fetch_dropping_odds(*, force: bool = False, max_age_minutes: int = 30) -> dict:
    if not force and DROPPING_CACHE.is_file():
        try:
            data = json.loads(DROPPING_CACHE.read_text(encoding="utf-8"))
            ts = data.get("fetched_at", "")
            if ts:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - fetched).total_seconds() / 60
                if age < max_age_minutes:
                    return {"ok": True, "n": len(data.get("rows", [])), "from_cache": True, "rows": data["rows"]}
        except Exception:
            pass
    try:
        html = _get_html(ODDSSAFARI_URL)
        rows = parse_oddssafari_dropping(html)
        payload = {"fetched_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
        RAW.mkdir(parents=True, exist_ok=True)
        DROPPING_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "n": len(rows), "from_cache": False, "rows": rows}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "n": 0, "rows": [], "from_cache": False}


def load_moneyway_cache() -> list[dict]:
    if not MONEYWAY_CACHE.is_file():
        return []
    try:
        data = json.loads(MONEYWAY_CACHE.read_text(encoding="utf-8"))
        fetched = data.get("fetched_at")
        rows = data.get("rows") or []
        if fetched:
            for r in rows:
                r.setdefault("_fetched_at", fetched)
        return rows
    except Exception:
        return []


def load_dropping_cache() -> list[dict]:
    if not DROPPING_CACHE.is_file():
        return []
    try:
        data = json.loads(DROPPING_CACHE.read_text(encoding="utf-8"))
        fetched = data.get("fetched_at")
        rows = data.get("rows") or []
        if fetched:
            for r in rows:
                r.setdefault("_fetched_at", fetched)
        return rows
    except Exception:
        return []


def lookup_moneyway(player_a: str, player_b: str, *, rows: list[dict] | None = None) -> dict | None:
    rows = rows if rows is not None else load_moneyway_cache()
    for row in rows:
        if _players_match(player_a, player_b, row["player_a"], row["player_b"]):
            return row
    return None


def lookup_dropping(player_a: str, player_b: str, *, pick_side: str | None = None, rows: list[dict] | None = None) -> dict | None:
    rows = rows if rows is not None else load_dropping_cache()
    for row in rows:
        if not _players_match(player_a, player_b, row["player_a"], row["player_b"]):
            continue
        if pick_side == "A" and row.get("side") != "1":
            if row.get("side") == "2":
                continue
        if pick_side == "B" and row.get("side") != "2":
            if row.get("side") == "1":
                continue
        return row
    return None


def sync_market_signals(*, force: bool = False) -> dict:
    mw = fetch_moneyway(force=force)
    drop = fetch_dropping_odds(force=force)
    return {
        "moneyway": {"ok": mw.get("ok"), "n": mw.get("n", 0), "from_cache": mw.get("from_cache"), "error": mw.get("error")},
        "dropping": {"ok": drop.get("ok"), "n": drop.get("n", 0), "from_cache": drop.get("from_cache"), "error": drop.get("error")},
    }
