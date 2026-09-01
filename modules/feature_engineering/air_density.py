"""Densità aria e CPI dinamico da altitudine + meteo (Open-Meteo)."""

from __future__ import annotations

import math

RHO_SEA_LEVEL = 1.225  # kg/m³ @ 15°C
R_DRY = 287.05


def vapor_pressure_pa(temp_c: float, humidity_pct: float) -> float:
    """Pressione di vapore saturo (Magnus, Pa)."""
    rh = max(0.0, min(100.0, float(humidity_pct)))
    t = float(temp_c)
    return (rh / 100.0) * 610.78 * math.exp(17.27 * t / (t + 237.3))


def air_density_kg_m3(
    *,
    temp_c: float,
    humidity_pct: float = 50.0,
    pressure_hpa: float | None = None,
    altitude_m: float = 0.0,
) -> float:
    """Densità aria (kg/m³) — temperatura, umidità, pressione o altitudine."""
    t_k = float(temp_c) + 273.15
    if pressure_hpa is not None and float(pressure_hpa) > 0:
        p_pa = float(pressure_hpa) * 100.0
    else:
        # Barometric formula ISA
        alt = max(0.0, float(altitude_m))
        p_pa = 101325.0 * (1.0 - 0.0065 * alt / 288.15) ** 5.255

    e = vapor_pressure_pa(temp_c, humidity_pct)
    return (p_pa - 0.378 * e) / (R_DRY * t_k)


def cpi_air_factor(
    rho: float,
    *,
    rho_ref: float = RHO_SEA_LEVEL,
    exponent: float = 0.22,
) -> float:
    """Aria meno densa → campo più veloce → fattore > 1."""
    rho = max(0.9, min(1.4, float(rho)))
    return (rho_ref / rho) ** exponent


def dynamic_cpi(
    cpi_nominal: float,
    *,
    weather: dict | None = None,
    altitude_m: float = 0.0,
) -> float:
    """CPI effettivo sessione (giorno/notte, caldo/freddo, quota)."""
    base = float(cpi_nominal or 1.0)
    if not weather and altitude_m <= 100:
        return base

    temp = weather.get("temp_c") if weather else 22.0
    hum = weather.get("humidity_pct") if weather else 50.0
    press = weather.get("pressure_hpa") if weather else None

    if temp is None:
        temp = 22.0
    if hum is None:
        hum = 50.0

    rho = air_density_kg_m3(
        temp_c=float(temp),
        humidity_pct=float(hum),
        pressure_hpa=float(press) if press is not None else None,
        altitude_m=float(altitude_m),
    )
    factor = cpi_air_factor(rho)
    return round(base * factor, 4)


def serve_adjustment_from_air(rho: float) -> float:
    """Delta P(serve) da densità aria (aria rarefatta favorisce servizio)."""
    factor = cpi_air_factor(rho)
    return round((factor - 1.0) * 0.08, 4)
