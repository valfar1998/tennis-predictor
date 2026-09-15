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
MIN_EDGE = 0.020
MIN_PROB_PLAY = 0.34
# EV: hard discard sopra questi cap; review nella fascia intermedia
EV_SANITY_CAP = 0.28
EV_SANITY_MAX_ODDS = 3.0
EV_SANITY_CAP_HIGH_ODDS = 0.25  # quota > SHARP_HIGH_ODDS_MIN → scarta se EV >
EV_SANITY_CAP_LOW_ODDS = 0.30  # quota <= SHARP_HIGH_ODDS_MIN → scarta se EV >
EV_REVIEW_THRESHOLD = 0.20  # EV > 20% → action review (non alert automatico)
MKT_DIVERGENCE_MAX = 0.20  # divergenza modello/mercato > 20% → no_bet
MKT_DIVERGENCE_SOFT = 0.12  # sopra questa soglia: riduci peso modello
ITF_BET_FREEZE = False  # sostituito da governance adattiva su BCR ITF
SHARP_HIGH_ODDS_MIN = 3.0
SHARP_ODDS_SOURCES = frozenset({"betfair", "pinnacle", "ps"})
BAYES_SHRINK_MIN_MATCHES = 10
BAYES_SHRINK_W_ITF = 0.22  # più peso modello su ITF (prima 0.15)
# Governance ITF adattiva (vedi itf_governance.py)
ITF_BCR_RELAX_THRESHOLD = 0.20
ITF_BCR_STRICT_THRESHOLD = 0.05
ITF_BCR_MIN_N = 10
ITF_SHRINK_W_RELAXED = 0.30
ITF_EV_CAP_RELAXED = 0.28
ITF_EV_CAP_STRICT = 0.24
ITF_MIN_DATA_DENSITY_STRICT = 18
# Ranking / giocabilità: preferisci edge su quote sostenibili
ODDS_VARIANCE_REF = 2.0  # quota di riferimento per Sharpe-like score
MISSING_SIGNAL_SCORE = 0.38  # MW/Drop assenti: meno punitivo (prima 0.32)
# Steam: solo dropping aggressivo blocca; variazioni fisiologiche passano
STEAM_EROSION_RATIO = 0.40  # era 0.55 (più severo)
STEAM_MIN_DROP_PCT = 8.0  # sotto questa soglia: no hard-block steam
STEAM_PLAYABILITY_SCORE = 0.28  # era 0.15
TOURNEY_LEVEL_CODE = {
    "G": 1.0,
    "M": 2.0,
    "F": 2.0,
    "A": 2.0,
    "D": 2.0,
    "C": 3.0,
    "S": 4.0,
}

# Risk controls (esecuzione)
CIRCUIT_BREAKER_MIN_EDGE = 0.028  # era 3.5% — stress meno cieco
DRAWDOWN_BREAKER_PCT = 0.20  # era 15%
STREAK_LOSS_UNITS = 11.0  # unità da 1% bankroll
UNIT_SIZE = 0.01
DAILY_EXPOSURE_CAP = 0.06  # 6% bankroll max stesso giorno/torneo
DAILY_EXPOSURE_MIN_BETS = 6

# Shadow bet (sample BCR sotto freeze / circuit breaker, Kelly=0)
SHADOW_BET_ENABLED = True
SHADOW_MIN_EDGE = 0.015
SHADOW_KELLY = 0.0
# Analisi tornei inferiori (Challenger/ITF): non scartare a priori
ANALYZE_LOWER_TIERS = True
LOWER_TIER_PARTIAL_RESOLVE_OK = True  # Betfair + 1 lato risolto → no hard-block

# BCR Betfair — qualità chiusura (esclude last-LTP ≈ quota bet)
BCR_ACTIONS = ("bet", "shadow")
BCR_QUALITY_CLOSE_SOURCES = frozenset(
    {
        "betfair_bsp",
        "betfair_t1",
        "betfair_t5",
        "betfair_t60",
        "betfair_settled",  # solo se non fallback (vedi live_metrics)
    }
)
BCR_FALLBACK_CLOSE_SOURCES = frozenset(
    {
        "betfair_ltp_fallback",
        "betfair_ltp",
        "betfair_bet_snapshot",
    }
)
BCR_MIN_CLOSE_DELTA = 0.01  # |odds_bet − close| sotto soglia → non è una chiusura
PREMATCH_CLOSE_WINDOWS_MIN = (60, 5, 1)

# Market signal temporal decay (T-15min ≈ 3× T-12h)
MARKET_DECAY_RATIO = 3.0
MARKET_DECAY_REF_HOURS = 12.0
MARKET_DECAY_NEAR_MINUTES = 15

# Serve/Return Elo
SR_ELO_SURFACE_WEIGHT = 0.68

# Retirement risk
RETIREMENT_MAX_P = 0.45

# Retirement rules per bookmaker (void policy)
RETIREMENT_RULES = {
    "pinnacle": "1_set",
    "bet365": "1_ball",
    "default": "1_set",
}
