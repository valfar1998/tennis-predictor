"""Date e filtri calendario (Europe/Rome, da oggi in avanti)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

CALENDAR_TZ = ZoneInfo("Europe/Rome")


def parse_commence_time(raw: Any) -> datetime | None:
    """Converte stringhe/API timestamp in datetime UTC-aware."""
    if raw is None or raw == "":
        return None

    dt: datetime | None = None
    if isinstance(raw, datetime):
        dt = raw
    elif hasattr(raw, "to_pydatetime"):
        try:
            dt = raw.to_pydatetime()
        except Exception:
            return None
    else:
        s = str(raw).strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def apply_local_schedule(pred: dict[str, Any], *, commence_time: Any = None) -> dict[str, Any]:
    """Allinea `date` e ora al fuso Europe/Rome (Betfair restituisce UTC)."""
    raw = commence_time or pred.get("commence_time_utc") or pred.get("_match_start_dt")
    dt_utc = parse_commence_time(raw)
    if dt_utc is None:
        day = str(pred.get("date") or "")[:10]
        if day:
            pred["date"] = day
        return pred

    local = dt_utc.astimezone(CALENDAR_TZ)
    pred["commence_time_utc"] = dt_utc.isoformat()
    pred["_match_start_dt"] = dt_utc.isoformat()
    pred["date"] = local.date().isoformat()
    pred["start_time_local"] = local.strftime("%H:%M")
    pred["timezone"] = "Europe/Rome"
    return pred


def calendar_day(pred: dict[str, Any]) -> date | None:
    start = parse_commence_time(pred.get("_match_start_dt") or pred.get("commence_time_utc"))
    if start is not None:
        return start.astimezone(CALENDAR_TZ).date()
    day = str(pred.get("date") or "")[:10]
    try:
        return date.fromisoformat(day)
    except ValueError:
        return None


def filter_from_today(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tieni solo match con data locale >= oggi (Europe/Rome)."""
    today = datetime.now(CALENDAR_TZ).date()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for pred in predictions:
        day = calendar_day(pred)
        if day is None or day < today:
            dropped += 1
            continue
        kept.append(pred)
    if dropped:
        print(f"  calendario: esclusi {dropped} match prima di oggi ({today.isoformat()})", flush=True)
    return kept


def normalize_predictions_calendar(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for pred in predictions:
        apply_local_schedule(pred)
    return filter_from_today(predictions)
