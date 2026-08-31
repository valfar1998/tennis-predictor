"""Costanti condivise per il sistema predittivo tennis."""

from __future__ import annotations

SURFACES = ("Hard", "Clay", "Grass", "Carpet")
SURFACE_ALIASES = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
    "indoor hard": "Hard",
    "outdoor hard": "Hard",
}

# Livello torneo → K-factor Elo
K_BY_LEVEL = {
    "G": 40,   # Grand Slam
    "M": 35,   # Masters 1000
    "A": 28,   # ATP tour
    "F": 32,   # Finals
    "C": 22,   # Challenger
    "S": 18,   # ITF/Satellite
    "D": 30,   # Davis Cup
}

DEFAULT_K = 28
ELO_START = 1500.0
ELO_SURFACE_WEIGHT = 0.68  # w in Rfinal = w*Rsurface + (1-w)*Rglobal
INACTIVITY_DAYS = 75
INACTIVITY_DECAY = 0.92

# Kelly
KELLY_FRACTION = 0.20
KELLY_CAP = 0.018
MIN_EDGE = 0.025
MIN_PROB_PLAY = 0.38

# Retirement rules per bookmaker (void policy)
RETIREMENT_RULES = {
    "pinnacle": "1_set",
    "bet365": "1_ball",
    "default": "1_set",
}
