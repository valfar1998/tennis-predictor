"""Avvisi Telegram value bet tennis (stile football-predictor)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from modules.notify.telegram import load_credentials, send_message, telegram_status

ROOT = Path(__file__).resolve().parents[2]
SENT = ROOT / "data" / "processed" / "telegram_alerts_sent.json"
KEEP_DAYS = 21
CHUNK = 8
BRAND = "TENNIS_PREDICTOR"
TZ = ZoneInfo("Europe/Rome")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def brand_header(*, continued: bool = False) -> str:
    when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    suffix = " (cont.)" if continued else ""
    return f"{BRAND} — alert scommesse{suffix}\n{when} Roma"


def _load_sent() -> dict[str, str]:
    if not SENT.is_file():
        return {}
    try:
        raw = json.loads(SENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def _save_sent(ids: dict[str, str]) -> None:
    cutoff = _now() - timedelta(days=KEEP_DAYS)
    kept: dict[str, str] = {}
    for key, ts in ids.items():
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            when = _now()
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= cutoff:
            kept[key] = ts
    SENT.parent.mkdir(parents=True, exist_ok=True)
    SENT.write_text(json.dumps(kept, indent=2), encoding="utf-8")


def alert_key(pred: dict) -> str:
    return "|".join(
        str(pred.get(k) or "")
        for k in ("player_a", "player_b", "date", "tourney", "betfair_event_id")
    )


def _format_bet(pred: dict) -> str:
    rec = pred.get("recommended") or {}
    date = str(pred.get("date") or "")[:10]
    tourney = str(pred.get("tourney") or "").strip()
    surface = str(pred.get("surface") or "")
    source = str(pred.get("odds_source") or "book")
    player_a = str(pred.get("player_a") or "?")
    player_b = str(pred.get("player_b") or "?")
    pick = str(rec.get("player") or "?")
    odds = rec.get("odds")
    p = rec.get("probability", 0)
    ev = rec.get("ev", 0)
    kelly = rec.get("kelly", 0)

    meta = " · ".join(x for x in (date, tourney) if x)
    lines = [
        meta or date,
        f"{player_a} vs {player_b}",
        f"Pick: {pick} @ {odds}",
        f"P={float(p):.1%} | EV={float(ev):+.1%} | Kelly={float(kelly):.2%}",
        f"Fonte quote: {source} | Superficie: {surface}",
    ]
    return "\n".join(lines)


def _pack(title: str, items: list[dict]) -> list[tuple[str, list[str]]]:
    if not items:
        return []
    out: list[tuple[str, list[str]]] = []
    for i in range(0, len(items), CHUNK):
        chunk = items[i : i + CHUNK]
        head = title if i == 0 else f"{title} (cont.)"
        body = "\n\n".join(_format_bet(p) for p in chunk)
        msg = f"{brand_header(continued=i > 0)}\n\n{head}\n\n{body}"
        out.append((msg, [alert_key(p) for p in chunk]))
    return out


def dispatch_alerts(predictions: list[dict] | None = None, *, dry_run: bool = False) -> dict:
    """Invia solo value bet nuovi (dedup su telegram_alerts_sent.json)."""
    rows = predictions or []
    bets = [p for p in rows if p.get("action") == "bet" and p.get("recommended")]
    sent_ids = _load_sent()
    fresh = [p for p in bets if alert_key(p) not in sent_ids]
    messages = _pack("🎯 VALUE BET · EV positivo", fresh)

    sent_n = 0
    if dry_run:
        for msg, _ids in messages:
            print(msg)
            print("---")
    elif messages:
        if not load_credentials():
            print("telegram skip: credenziali assenti")
        else:
            now = _now().isoformat()
            changed = False
            for msg, ids in messages:
                if send_message(msg):
                    sent_n += 1
                    for key in ids:
                        sent_ids[key] = now
                    changed = True
            if changed:
                _save_sent(sent_ids)

    info = {
        "n_bets": len(bets),
        "n_new_bets": len(fresh),
        "n_messages": len(messages),
        "n_sent": sent_n,
        "dry_run": dry_run,
        "status": telegram_status(),
    }
    print(
        f"telegram avvisi: value {info['n_new_bets']}/{info['n_bets']} nuovi, "
        f"inviati {sent_n}"
    )
    return info
