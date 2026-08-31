"""Tabella statica altitudine sedi torneo (metri). Fonte: dati pubblici ATP/WTA."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALT_PATH = ROOT / "data" / "raw" / "venue_altitude.json"

# Altitudine approssimativa in metri — tornei noti ad alta quota
DEFAULT_ALTITUDES: dict[str, float] = {
    "madrid": 650,
    "gstaad": 1050,
    "kitzbuhel": 762,
    "kitzbühel": 762,
    "bogota": 2640,
    "bogotá": 2640,
    "quito": 2850,
    "mexico city": 2240,
    "denver": 1609,
    "salt lake city": 1288,
    "johannesburg": 1753,
    "johannesbourg": 1753,
    "santiago": 520,
    "cali": 1000,
    "medellin": 1495,
    "medellín": 1495,
    "asuncion": 130,
    "asunción": 130,
    "rome": 21,
    "roma": 21,
    "monte carlo": 12,
    "paris": 35,
    "london": 11,
    "wimbledon": 11,
    "new york": 10,
    "melbourne": 31,
    "sydney": 58,
    "beijing": 44,
    "shanghai": 4,
    "tokyo": 40,
    "dubai": 5,
    "doha": 7,
    "cincinnati": 147,
    "atlanta": 320,
    "winston-salem": 280,
    "stuttgart": 245,
    "halle": 116,
    "queens club": 11,
    "barcelona": 12,
    "hamburg": 6,
    "rotterdam": -1,
    "acapulco": 30,
    "rio de janeiro": 11,
    "buenos aires": 25,
    "sao paulo": 760,
    "são paulo": 760,
}


def _norm_key(name: str) -> str:
    return str(name or "").strip().lower()


def load_altitudes() -> dict[str, float]:
    if ALT_PATH.exists():
        try:
            data = json.loads(ALT_PATH.read_text(encoding="utf-8"))
            return {k.lower(): float(v) for k, v in data.items()}
        except Exception:
            pass
    ALT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALT_PATH.write_text(json.dumps(DEFAULT_ALTITUDES, indent=2), encoding="utf-8")
    return dict(DEFAULT_ALTITUDES)


def lookup_altitude(tourney_name: str | None, city: str | None = None) -> float:
    """Restituisce altitudine in metri (0 = livello del mare default)."""
    alts = load_altitudes()
    for key in (_norm_key(tourney_name), _norm_key(city)):
        if not key:
            continue
        if key in alts:
            return alts[key]
        for name, alt in alts.items():
            if name in key or key in name:
                return alt
    return 0.0


def altitude_serve_boost(altitude_m: float) -> float:
    """Fattore correttivo per P(serve win) in alta quota (aria meno densa)."""
    if altitude_m <= 100:
        return 0.0
    # ~0.5% per 100m sopra 500m
    return min(0.04, max(0.0, (altitude_m - 500) / 100 * 0.005))
