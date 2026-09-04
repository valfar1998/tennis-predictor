"""Cache quote di chiusura OddsPortal (popolata da job async / script opzionale)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from modules.data_update.entity_resolution import _last_name, odds_match_key

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "oddsportal_close.json"


def _load() -> dict:
    if not CACHE.exists():
        return {"rows": [], "updated_at": None}
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {"rows": [], "updated_at": None}


def save_close_rows(rows: list[dict]) -> dict:
    """Salva righe {date, winner, loser, psw, psl} da scraper esterno."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
    CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "n": len(rows), "path": str(CACHE)}


def lookup_oddsportal_close(
    player_a: str,
    player_b: str,
    *,
    date: str | None = None,
) -> dict | None:
    """Legge chiusura Pinnacle/bookmaker da cache OddsPortal (se presente)."""
    data = _load()
    day = str(date or "")[:10]
    key = odds_match_key(day, player_a, player_b) if day else None
    rev = odds_match_key(day, player_b, player_a) if day else None
    fallback = None
    for row in data.get("rows") or []:
        row_day = str(row.get("date") or "")[:10]
        w, l = str(row.get("winner") or ""), str(row.get("loser") or "")
        psw, psl = row.get("psw"), row.get("psl")
        if not psw or not psl:
            continue
        name_ok = (
            _last_name(w) in (_last_name(player_a), _last_name(player_b))
            and _last_name(l) in (_last_name(player_a), _last_name(player_b))
            and _last_name(w) != _last_name(l)
        )
        if not name_ok:
            continue
        mapped = (
            {"a": float(psw), "b": float(psl), "source": "oddsportal"}
            if _last_name(w) == _last_name(player_a)
            else {"a": float(psl), "b": float(psw), "source": "oddsportal"}
        )
        if day and row_day:
            rk = odds_match_key(row_day, w, l)
            rkr = odds_match_key(row_day, l, w)
            if key and (rk in (key, rev) or rkr in (key, rev)):
                return mapped
            continue
        if day and not row_day:
            # riga senza data: candidate fallback se nomi matchano
            fallback = fallback or mapped
            continue
        if not day:
            return mapped
    return fallback


def row_count() -> int:
    return len(_load().get("rows") or [])
