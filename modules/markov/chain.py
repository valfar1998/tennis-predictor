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
    p_b_return = 1.0 - p_b
    first = 0 if first_server_a else 1
    max_pts = 28  # copre tiebreak lunghi (es. 15-13) senza ricorsione profonda
    memo: dict[tuple[int, int], float] = {}

    def server_at(n: int) -> int:
        if n == 0:
            return first
        if n == 1:
            return 1 - first
        k = (n - 1) // 2
        return (1 - first) if k % 2 == 0 else first

    for a_pts in range(max_pts + 1):
        for b_pts in range(max_pts + 1):
            if a_pts >= 7 and a_pts - b_pts >= 2:
                memo[(a_pts, b_pts)] = 1.0
            elif b_pts >= 7 and b_pts - a_pts >= 2:
                memo[(a_pts, b_pts)] = 0.0

    for total in range(2 * max_pts, -1, -1):
        for a_pts in range(max(0, total - max_pts), min(total, max_pts) + 1):
            b_pts = total - a_pts
            key = (a_pts, b_pts)
            if key in memo:
                continue
            srv = server_at(a_pts + b_pts)
            p_win_pt = p_a if srv == 0 else p_b_return
            p_win = memo.get((a_pts + 1, b_pts), 0.5)
            p_lose = memo.get((a_pts, b_pts + 1), 0.5)
            memo[key] = p_win_pt * p_win + (1 - p_win_pt) * p_lose

    return memo.get((0, 0), 0.5)


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
