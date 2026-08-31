"""Distanza tra tornei (Haversine) e stima jet lag."""

from __future__ import annotations

import math
import re

# Coordinate approssimative venue ATP/WTA principali
VENUE_COORDS: dict[str, tuple[float, float]] = {
    "melbourne": (-37.81, 144.96),
    "australian open": (-37.81, 144.96),
    "roland garros": (48.85, 2.25),
    "paris": (48.85, 2.25),
    "wimbledon": (51.43, -0.21),
    "london": (51.43, -0.21),
    "us open": (40.75, -73.85),
    "new york": (40.75, -73.85),
    "indian wells": (33.72, -116.32),
    "miami": (25.76, -80.19),
    "madrid": (40.42, -3.70),
    "rome": (41.93, 12.46),
    "monte carlo": (43.74, 7.42),
    "barcelona": (41.39, 2.17),
    "cincinnati": (39.10, -84.51),
    "toronto": (43.65, -79.38),
    "montreal": (45.50, -73.57),
    "shanghai": (31.20, 121.50),
    "beijing": (39.99, 116.33),
    "tokyo": (35.68, 139.69),
    "dubai": (25.20, 55.27),
    "doha": (25.29, 51.53),
    "halle": (52.04, 8.34),
    "queens": (51.48, -0.21),
    "stuttgart": (48.78, 9.18),
    "atp finals": (51.50, -0.12),
    "basel": (47.56, 7.59),
    "vienna": (48.21, 16.37),
    "paris masters": (48.84, 2.28),
}


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(name or "").lower()).strip()


def lookup_coords(tourney_name: str | None) -> tuple[float, float] | None:
    key = _norm_key(tourney_name or "")
    if not key:
        return None
    for token, coords in VENUE_COORDS.items():
        if token in key:
            return coords
    return None


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def travel_km(from_tourney: str | None, to_tourney: str | None) -> float | None:
    a = lookup_coords(from_tourney)
    b = lookup_coords(to_tourney)
    if not a or not b:
        return None
    return round(haversine_km(a, b), 1)


def timezone_shift_hours(from_tourney: str | None, to_tourney: str | None) -> float | None:
    a = lookup_coords(from_tourney)
    b = lookup_coords(to_tourney)
    if not a or not b:
        return None
    return round((b[1] - a[1]) / 15.0, 1)
