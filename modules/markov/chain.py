"""Modello Markov gerarchico: punto → game → tiebreak → set → match.

Basato su Barnett & Clarke / O'Malley.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def game_win_prob(p: float) -> float:
    """P(vincere il game al servizio) dato P(vincere un punto al servizio)."""
    p = max(0.01, min(0.99, p))
    num = p**4 * (15 - 34 * p + 28 * p**2 - 8 * p**3)
    den = 1 - 2 * p * (1 - p)
    return num / den if den > 1e-9 else p


@lru_cache(maxsize=4096)
def tiebreak_win_prob(p_a: float, p_b: float, first_server_a: bool = True) -> float:
    """P(A vince tiebreak) con sequenza ABBA, p_a = P(A vince punto al servizio)."""
    p_a = max(0.01, min(0.99, p_a))
    p_b = max(0.01, min(0.99, p_b))
    p_b_return = 1.0 - p_b  # P(A vince punto in risposta quando B serve)

    memo: dict[tuple, float] = {}

    def dp(a_pts: int, b_pts: int, server: int, streak: int) -> float:
        """server: 0=A serve, 1=B serve. streak: punti consecutivi stesso server."""
        if a_pts >= 7 and a_pts - b_pts >= 2:
            return 1.0
        if b_pts >= 7 and b_pts - a_pts >= 2:
            return 0.0
        key = (a_pts, b_pts, server, streak)
        if key in memo:
            return memo[key]

        if server == 0:
            p_win_pt = p_a
        else:
            p_win_pt = p_b_return

        # ABBA: dopo 1 punto cambia server, dopo altri 2 cambia di nuovo
        if streak == 0:
            next_streak = 1
            next_server = server
        elif streak == 1:
            next_streak = 0
            next_server = 1 - server
        else:
            next_streak = 0
            next_server = 1 - server

        p_if_win = dp(a_pts + 1, b_pts, next_server, next_streak) if server == 0 or True else dp(a_pts + 1, b_pts, next_server, next_streak)
        if server == 0:
            p_win = dp(a_pts + 1, b_pts, next_server, next_streak)
            p_lose = dp(a_pts, b_pts + 1, next_server, next_streak)
        else:
            p_win = dp(a_pts + 1, b_pts, next_server, next_streak)
            p_lose = dp(a_pts, b_pts + 1, next_server, next_streak)

        result = p_win_pt * p_win + (1 - p_win_pt) * p_lose
        memo[key] = result
        return result

    start_server = 0 if first_server_a else 1
    return dp(0, 0, start_server, 0)


def set_win_prob(p_hold_a: float, p_hold_b: float, *, first_server_a: bool = True) -> float:
    """P(A vince set) con probabilità di tenuta servizio per entrambi."""
    memo: dict[tuple, float] = {}

    def dp(ga: int, gb: int, server_a: bool) -> float:
        if ga >= 6 and ga - gb >= 2:
            return 1.0
        if gb >= 6 and gb - ga >= 2:
            return 0.0
        if ga == 6 and gb == 6:
            return tiebreak_win_prob(p_hold_a, p_hold_b, first_server_a=server_a)

        key = (ga, gb, server_a)
        if key in memo:
            return memo[key]

        if server_a:
            p_hold = p_hold_a
            p_win_set = dp(ga + 1, gb, False)
            p_lose = dp(ga, gb + 1, False)
        else:
            p_hold = 1.0 - p_hold_b
            p_win_set = dp(ga + 1, gb, True)
            p_lose = dp(ga, gb + 1, True)

        result = p_hold * p_win_set + (1 - p_hold) * p_lose
        memo[key] = result
        return result

    return dp(0, 0, first_server_a)


def match_win_prob(
    p_serve_a: float,
    p_serve_b: float,
    *,
    best_of: int = 3,
    first_server_a: bool = True,
) -> float:
    """P(A vince match) BO3 o BO5."""
    p_hold_a = game_win_prob(p_serve_a)
    p_hold_b = game_win_prob(p_serve_b)
    p_set = set_win_prob(p_hold_a, p_hold_b, first_server_a=first_server_a)
    sets_to_win = (best_of + 1) // 2

    memo: dict[tuple, float] = {}

    def dp(sa: int, sb: int) -> float:
        if sa >= sets_to_win:
            return 1.0
        if sb >= sets_to_win:
            return 0.0
        key = (sa, sb)
        if key in memo:
            return memo[key]
        result = p_set * dp(sa + 1, sb) + (1 - p_set) * dp(sa, sb + 1)
        memo[key] = result
        return result

    return dp(0, 0)


def estimate_serve_probs(
    elo_a: float,
    elo_b: float,
    *,
    surface_wr_a: float = 0.5,
    surface_wr_b: float = 0.5,
    base_serve: float = 0.62,
    elo_scale: float = 0.0008,
    adjustments: float = 0.0,
) -> tuple[float, float]:
    """Stima P(serve win) da Elo e statistiche di superficie."""
    elo_diff = elo_a - elo_b
    p_a = base_serve + elo_scale * elo_diff + 0.04 * (surface_wr_a - 0.5) + adjustments
    p_b = base_serve - elo_scale * elo_diff + 0.04 * (surface_wr_b - 0.5) - adjustments
    return max(0.45, min(0.78, p_a)), max(0.45, min(0.78, p_b))
