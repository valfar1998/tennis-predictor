"""Transizione superficie: riduce peso Elo surface nei primi 7 giorni di nuova stagione."""

from __future__ import annotations

from datetime import date, datetime

from modules.constants import ELO_SURFACE_WEIGHT, SURFACE_TRANSITION_DAYS

# Inizio effettivo stagione per superficie (ATP/WTA, approssimazione calendario)
SURFACE_SEASON_START: dict[str, tuple[int, int]] = {
    "Clay": (4, 7),    # Monte Carlo / stagione terra
    "Grass": (6, 10),  # Stuttgart / Queen's
    "Hard": (12, 29),  # offseason indoor → Australian summer (dic-gen)
    # Hard outdoor US: trattato come continuazione Hard; secondo anchor estivo opzionale
}

# Secondo anchor Hard outdoor (post-Wimbledon)
HARD_OUTDOOR_START = (8, 1)


def _as_date(d: date | datetime | str | None) -> date:
    if d is None:
        return date.today()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def _days_since_anchor(as_of: date, anchor: tuple[int, int]) -> int:
    month, day = anchor
    year = as_of.year
    anchor_date = date(year, month, day)
    if as_of < anchor_date:
        anchor_date = date(year - 1, month, day)
    return max(0, (as_of - anchor_date).days)


def days_into_surface_season(surface: str, as_of: date | datetime | str | None = None) -> int:
    """Giorni trascorsi dall'inizio della stagione corrente per la superficie."""
    surf = str(surface or "Hard").title()
    as_of_d = _as_date(as_of)

    if surf == "Hard":
        # Prendere l'anchor più recente tra US hard e indoor/AO
        d_us = _days_since_anchor(as_of_d, HARD_OUTDOOR_START)
        d_winter = _days_since_anchor(as_of_d, SURFACE_SEASON_START["Hard"])
        return min(d_us, d_winter)

    anchor = SURFACE_SEASON_START.get(surf)
    if not anchor:
        return SURFACE_TRANSITION_DAYS
    return _days_since_anchor(as_of_d, anchor)


def transition_weight_multiplier(
    surface: str,
    as_of: date | datetime | str | None = None,
    *,
    transition_days: int | None = None,
) -> float:
    """1.0 = peso pieno; ~0.5 all'inizio transizione (più peso Elo global)."""
    td = transition_days if transition_days is not None else SURFACE_TRANSITION_DAYS
    days = days_into_surface_season(surface, as_of)
    if days >= td:
        return 1.0
    floor = 0.50
    return floor + (1.0 - floor) * (days / td)


def transition_surface_weight(
    base_weight: float | None = None,
    surface: str = "Hard",
    as_of: date | datetime | str | None = None,
) -> float:
    """Peso Elo surface effettivo in fase live (CPI applicato separatamente)."""
    w = base_weight if base_weight is not None else ELO_SURFACE_WEIGHT
    return w * transition_weight_multiplier(surface, as_of)


def transition_context(surface: str, as_of: date | datetime | str | None = None) -> dict:
    """Metadati per audit predizione live."""
    mult = transition_weight_multiplier(surface, as_of)
    days = days_into_surface_season(surface, as_of)
    return {
        "surface": surface,
        "days_into_season": days,
        "weight_multiplier": round(mult, 3),
        "in_transition": mult < 1.0,
    }
