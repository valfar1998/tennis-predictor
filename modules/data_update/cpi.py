"""CPI unificato: CourtSpeed (primario) + UTS (fallback)."""

from __future__ import annotations

from modules.data_update.courtspeed import cpi_serve_adjustment as _cs_adj
from modules.data_update.courtspeed import lookup_courtspeed_cpi
from modules.data_update.uts import cpi_serve_adjustment as _uts_adj
from modules.data_update.uts import lookup_cpi as lookup_uts_cpi


def lookup_cpi(tourney_name: str, *, surface: str | None = None) -> float | None:
    """Restituisce CPI normalizzato (~30-45 scala CourtSpeed, convertito a ~0.9-1.1)."""
    raw = lookup_courtspeed_cpi(tourney_name, surface=surface)
    if raw is not None:
        return _normalize_courtspeed(raw)
    uts = lookup_uts_cpi(tourney_name)
    return uts


def _normalize_courtspeed(cpi: float) -> float:
    """Converte CPI CourtSpeed (30=slow, 45=fast) in scala ~1.0 neutra."""
    return round(0.85 + (cpi - 30) / 100, 3)


def effective_cpi(
    tourney_name: str,
    *,
    surface: str | None = None,
    weather: dict | None = None,
    altitude_m: float | None = None,
) -> float:
    """CPI nominale modulato da densità aria (meteo + altitudine)."""
    from modules.data_update.altitude import lookup_altitude
    from modules.feature_engineering.air_density import dynamic_cpi

    nominal = lookup_cpi(tourney_name, surface=surface) or 1.0
    alt = float(altitude_m) if altitude_m is not None else lookup_altitude(tourney_name)
    return dynamic_cpi(nominal, weather=weather, altitude_m=alt)


def cpi_serve_adjustment(cpi: float | None) -> float:
    if cpi is None:
        return 0.0
    if cpi > 2:
        return _cs_adj(cpi)
    return _uts_adj(cpi)
