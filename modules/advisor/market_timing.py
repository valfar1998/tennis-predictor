"""Pesatura temporale segnali mercato (sharp money vs flussi vecchi)."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

# T-15 min pesa ~3× rispetto a T-12 ore
_REF_MINUTES = 12 * 60
_NEAR_MINUTES = 15
_RATIO_NEAR_VS_REF = 3.0
_LAMBDA = math.log(_RATIO_NEAR_VS_REF) / max(_REF_MINUTES - _NEAR_MINUTES, 1)


def temporal_weight(minutes_to_start: float | None) -> float:
    """Peso esponenziale: più vicino all'inizio → più peso."""
    if minutes_to_start is None:
        return 0.55
    m = max(0.0, float(minutes_to_start))
    if m <= _NEAR_MINUTES:
        return 1.0
    return float(min(1.0, math.exp(-_LAMBDA * (m - _NEAR_MINUTES))))


def _parse_datetime_label(label: str, *, year: int | None = None) -> datetime | None:
    """Parse etichette Arbworld/OddsSafari."""
    if not label:
        return None
    s = str(label).strip()
    yr = year or datetime.now(timezone.utc).year

    m = re.match(r"([A-Za-z]{3})\s+(\d{1,2}),\s*(\d{2}):(\d{2})", s)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {yr} {m.group(3)}:{m.group(4)}", "%b %d %Y %H:%M")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    m2 = re.match(r"(\d{2})/(\d{2})\s+(\d{2}):(\d{2})", s)
    if m2:
        try:
            dt = datetime.strptime(f"{m2.group(1)}/{m2.group(2)}/{yr} {m2.group(3)}:{m2.group(4)}", "%d/%m/%Y %H:%M")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def minutes_until_match(
    row: dict,
    *,
    match_start: datetime | None = None,
    fetched_at: datetime | None = None,
) -> float | None:
    """Minuti mancanti all'inizio (da etichetta match o commence_time)."""
    now = fetched_at or datetime.now(timezone.utc)

    if match_start is not None:
        if match_start.tzinfo is None:
            match_start = match_start.replace(tzinfo=timezone.utc)
        return (match_start - now).total_seconds() / 60.0

    for key in ("commence_time", "match_start", "start_time"):
        raw = row.get(key)
        if raw:
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return (dt - now).total_seconds() / 60.0
            except ValueError:
                pass

    label = row.get("date_label") or row.get("datetime_label") or ""
    parsed = _parse_datetime_label(str(label))
    if parsed:
        return (parsed - now).total_seconds() / 60.0
    return None


def apply_temporal_weight(score: float, minutes_to_start: float | None) -> tuple[float, float]:
    """Applica decadimento; ritorna (score_pesato, peso)."""
    w = temporal_weight(minutes_to_start)
    return round(float(score) * w, 4), round(w, 4)
