"""Meteo pre-match da Open-Meteo (gratuito, senza API key)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
GEO_CACHE = ROOT / "data" / "raw" / "geocode_cache.json"
WX_CACHE = ROOT / "data" / "raw" / "weather_cache.json"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_json(url: str, timeout: int = 12) -> dict:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode_city(city: str) -> dict | None:
    key = " ".join(str(city or "").strip().lower().split())
    if not key or len(key) < 2:
        return None
    cache = _load(GEO_CACHE)
    if key in cache:
        return cache[key]
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
            {"name": city.strip(), "count": 1, "language": "en"}
        )
        data = _get_json(url)
        results = data.get("results") or []
        if not results:
            return None
        hit = results[0]
        row = {
            "lat": float(hit["latitude"]),
            "lon": float(hit["longitude"]),
            "name": hit.get("name") or city,
            "country": hit.get("country") or "",
        }
        cache[key] = row
        _save(GEO_CACHE, cache)
        return row
    except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError):
        return None


def fetch_weather(lat: float, lon: float, when: datetime | str) -> dict | None:
    """Forecast or historical weather at match time."""
    from modules.data_update.calendar_utils import parse_commence_time

    when_dt = parse_commence_time(when)
    if when_dt is None:
        return None

    cache_key = f"{lat:.3f},{lon:.3f},{when_dt.strftime('%Y-%m-%d')}"
    cache = _load(WX_CACHE)
    if cache_key in cache:
        return cache[cache_key]

    date_str = when_dt.strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc)
    is_past = when_dt < now

    try:
        if is_past:
            url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode({
                "latitude": lat, "longitude": lon,
                "start_date": date_str, "end_date": date_str,
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
                "timezone": "auto",
            })
        else:
            url = "https://api.open-meteo.com/v1/forecast?" + urlencode({
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
                "timezone": "auto",
                "forecast_days": 7,
            })
        data = _get_json(url)
        hourly = data.get("hourly") or {}
        temps = hourly.get("temperature_2m") or []
        hums = hourly.get("relative_humidity_2m") or []
        press = hourly.get("surface_pressure") or []
        winds = hourly.get("wind_speed_10m") or []
        if not temps:
            return None
        mid = len(temps) // 2
        row = {
            "temp_c": temps[mid] if mid < len(temps) else temps[0],
            "humidity_pct": hums[mid] if mid < len(hums) else None,
            "pressure_hpa": press[mid] if mid < len(press) else None,
            "wind_kmh": winds[mid] if mid < len(winds) else None,
        }
        cache[cache_key] = row
        _save(WX_CACHE, cache)
        return row
    except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError):
        return None


def weather_serve_adjustment(weather: dict | None) -> float:
    """Ajust P(serve) per condizioni: aria calda/secca favorisce il servizio."""
    if not weather:
        return 0.0
    adj = 0.0
    temp = weather.get("temp_c")
    hum = weather.get("humidity_pct")
    if temp is not None and temp > 28:
        adj += min(0.02, (temp - 28) * 0.002)
    if hum is not None and hum < 40:
        adj += 0.005
    wind = weather.get("wind_kmh")
    if wind is not None and wind > 25:
        adj -= min(0.015, (wind - 25) * 0.001)
    return adj
