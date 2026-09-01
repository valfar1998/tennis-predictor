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
SURFACE_TRANSITION_DAYS = 7

MCP_MIN_MATCHES = 5
MCP_PRIOR_WEIGHT = 5.0
DATASET_LOW_SAMPLE = 15
DATASET_PRIOR_WEIGHT = 15.0
CIRCUIT_BP_SAVE = 0.635
CIRCUIT_TB_CLUTCH = 0.640

# Kelly
KELLY_FRACTION = 0.20
KELLY_CAP = 0.018
KELLY_CAP_BY_LEVEL = {
    "G": 0.018,  # Grand Slam
    "M": 0.018,  # Masters 1000
    "F": 0.018,  # Finals
    "A": 0.012,  # ATP/WTA tour (250/500)
    "D": 0.015,  # Davis Cup
    "C": 0.010,  # Challenger
    "S": 0.008,  # ITF
}
MIN_EDGE = 0.025
MIN_PROB_PLAY = 0.38
EV_SANITY_CAP = 0.35
EV_SANITY_MAX_ODDS = 3.0

# Risk controls (esecuzione)
CIRCUIT_BREAKER_MIN_EDGE = 0.045
DRAWDOWN_BREAKER_PCT = 0.15
STREAK_LOSS_UNITS = 11.0  # unità da 1% bankroll
UNIT_SIZE = 0.01
DAILY_EXPOSURE_CAP = 0.06  # 6% bankroll max stesso giorno/torneo
DAILY_EXPOSURE_MIN_BETS = 6

# Market signal temporal decay (T-15min ≈ 3× T-12h)
MARKET_DECAY_RATIO = 3.0
MARKET_DECAY_REF_HOURS = 12.0
MARKET_DECAY_NEAR_MINUTES = 15

# Serve/Return Elo
SR_ELO_SURFACE_WEIGHT = 0.68

# Retirement risk
RETIREMENT_MAX_P = 0.38

# Retirement rules per bookmaker (void policy)
RETIREMENT_RULES = {
    "pinnacle": "1_set",
    "bet365": "1_ball",
    "default": "1_set",
}
