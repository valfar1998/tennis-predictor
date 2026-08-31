"""Download quote storiche da tennis-data.co.uk (gratuito).

Wrapper retrocompatibile — logica completa in tennis_data_portal.py.
"""

from __future__ import annotations

from modules.data_update.tennis_data_portal import (
    download_atp_seasons,
    load_odds_all,
    load_odds_atp,
    load_odds_wta,
    sync_tennis_data_portal,
)


def download_tennis_data_odds(*, force: bool = False) -> dict:
    """Scarica stagioni ATP + tornei prioritari ATP/WTA."""
    return sync_tennis_data_portal(force=force)


def load_odds_history(*, tour: str | None = None):
    if tour and tour.upper() == "WTA":
        return load_odds_wta()
    if tour and tour.upper() == "ATP":
        return load_odds_atp()
    return load_odds_all()
