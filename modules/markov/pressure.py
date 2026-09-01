"""Markov con P(serve) dinamica: break point, tiebreak clutch + shrinkage MCP."""

from __future__ import annotations

from modules.markov.chain import (
    estimate_serve_probs,
    game_win_prob,
    match_win_prob,
    set_win_prob,
    tiebreak_win_prob,
)

from modules.constants import (
    CIRCUIT_BP_SAVE,
    CIRCUIT_TB_CLUTCH,
    DATASET_LOW_SAMPLE,
    DATASET_PRIOR_WEIGHT,
    MCP_MIN_MATCHES,
    MCP_PRIOR_WEIGHT,
)


def _clamp(p: float, lo: float = 0.45, hi: float = 0.78) -> float:
    return max(lo, min(hi, p))


def _adaptive_prior_weight(n_matches: int) -> float:
    if int(n_matches or 0) < DATASET_LOW_SAMPLE:
        return DATASET_PRIOR_WEIGHT
    return MCP_PRIOR_WEIGHT


def bayesian_shrink(
    observed: float | None,
    circuit_mean: float,
    n_matches: int,
    *,
    min_matches: int = MCP_MIN_MATCHES,
    prior_weight: float | None = None,
) -> float:
    """Regolarizza verso la media circuito se pochi match (MCP o dataset < 15)."""
    if observed is None:
        return circuit_mean
    n = max(0, int(n_matches or 0))
    threshold = max(min_matches, DATASET_LOW_SAMPLE)
    if n >= threshold:
        return observed
    k = prior_weight if prior_weight is not None else _adaptive_prior_weight(n)
    return (n * observed + k * circuit_mean) / (n + k)


def dynamic_serve_probs(
    p_serve_a: float,
    p_serve_b: float,
    *,
    bp_save_a: float | None = None,
    bp_save_b: float | None = None,
    tb_clutch_a: float | None = None,
    tb_clutch_b: float | None = None,
    n_matches_a: int = 0,
    n_matches_b: int = 0,
) -> tuple[float, float, float, float, float, float]:
    """Restituisce P(serve) standard e sotto pressione per A e B."""
    bp_a = bayesian_shrink(bp_save_a, CIRCUIT_BP_SAVE, n_matches_a) if bp_save_a is not None else (
        bayesian_shrink(None, CIRCUIT_BP_SAVE, n_matches_a) if n_matches_a < MCP_MIN_MATCHES else p_serve_a
    )
    bp_b = bayesian_shrink(bp_save_b, CIRCUIT_BP_SAVE, n_matches_b) if bp_save_b is not None else (
        bayesian_shrink(None, CIRCUIT_BP_SAVE, n_matches_b) if n_matches_b < MCP_MIN_MATCHES else p_serve_b
    )
    tb_a = bayesian_shrink(tb_clutch_a, CIRCUIT_TB_CLUTCH, n_matches_a) if tb_clutch_a is not None else p_serve_a
    tb_b = bayesian_shrink(tb_clutch_b, CIRCUIT_TB_CLUTCH, n_matches_b) if tb_clutch_b is not None else p_serve_b

    p_bp_a = _clamp(0.65 * p_serve_a + 0.35 * bp_a)
    p_bp_b = _clamp(0.65 * p_serve_b + 0.35 * bp_b)
    p_tb_a = _clamp(0.55 * p_serve_a + 0.45 * tb_a)
    p_tb_b = _clamp(0.55 * p_serve_b + 0.45 * tb_b)
    return p_serve_a, p_serve_b, p_bp_a, p_bp_b, p_tb_a, p_tb_b

def match_win_prob_pressure(
    p_serve_a: float,
    p_serve_b: float,
    *,
    best_of: int = 3,
    first_server_a: bool = True,
    bp_save_a: float | None = None,
    bp_save_b: float | None = None,
    tb_clutch_a: float | None = None,
    tb_clutch_b: float | None = None,
    n_matches_a: int = 0,
    n_matches_b: int = 0,
    bp_game_weight: float = 0.12,
) -> float:
    """P(A vince) con hold standard + mix BP e tiebreak clutch."""
    p_a, p_b, p_bp_a, p_bp_b, p_tb_a, p_tb_b = dynamic_serve_probs(
        p_serve_a,
        p_serve_b,
        bp_save_a=bp_save_a,
        bp_save_b=bp_save_b,
        tb_clutch_a=tb_clutch_a,
        tb_clutch_b=tb_clutch_b,
        n_matches_a=n_matches_a,
        n_matches_b=n_matches_b,
    )

    hold_a = game_win_prob(p_a)
    hold_b = game_win_prob(p_b)
    hold_bp_a = game_win_prob(p_bp_a)
    hold_bp_b = game_win_prob(p_bp_b)

    eff_hold_a = (1 - bp_game_weight) * hold_a + bp_game_weight * hold_bp_a
    eff_hold_b = (1 - bp_game_weight) * hold_b + bp_game_weight * hold_bp_b

    p_set = _set_win_prob_clutch(
        eff_hold_a,
        eff_hold_b,
        p_tb_a,
        p_tb_b,
        first_server_a=first_server_a,
    )

    sets_to_win = (best_of + 1) // 2
    memo: dict[tuple[int, int], float] = {}

    def dp(sa: int, sb: int) -> float:
        if sa >= sets_to_win:
            return 1.0
        if sb >= sets_to_win:
            return 0.0
        key = (sa, sb)
        if key in memo:
            return memo[key]
        memo[key] = p_set * dp(sa + 1, sb) + (1 - p_set) * dp(sa, sb + 1)
        return memo[key]

    return dp(0, 0)


def _set_win_prob_clutch(
    p_hold_a: float,
    p_hold_b: float,
    p_tb_a: float,
    p_tb_b: float,
    *,
    first_server_a: bool,
) -> float:
    """Set win con tiebreak che usa hold clutch dedicati."""
    memo: dict[tuple, float] = {}

    def dp(ga: int, gb: int, server_a: bool) -> float:
        if ga >= 6 and ga - gb >= 2:
            return 1.0
        if gb >= 6 and gb - ga >= 2:
            return 0.0
        if ga == 6 and gb == 6:
            return tiebreak_win_prob(p_tb_a, p_tb_b, first_server_a=server_a)

        key = (ga, gb, server_a)
        if key in memo:
            return memo[key]

        if server_a:
            p_hold = p_hold_a
            p_win = dp(ga + 1, gb, False)
            p_lose = dp(ga, gb + 1, False)
        else:
            p_hold = 1.0 - p_hold_b
            p_win = dp(ga + 1, gb, True)
            p_lose = dp(ga, gb + 1, True)

        memo[key] = p_hold * p_win + (1 - p_hold) * p_lose
        return memo[key]

    return dp(0, 0, first_server_a)


def estimate_serve_probs_full(
    elo_a: float,
    elo_b: float,
    *,
    surface_wr_a: float = 0.5,
    surface_wr_b: float = 0.5,
    adjustments: float = 0.0,
    pressure_a: dict | None = None,
    pressure_b: dict | None = None,
    best_of: int = 3,
    serve_elo_a: float | None = None,
    return_elo_a: float | None = None,
    serve_elo_b: float | None = None,
    return_elo_b: float | None = None,
    cpi_factor: float = 1.0,
) -> dict:
    """Pipeline Markov completa con profili pressione MCP."""
    p_serve_a, p_serve_b = estimate_serve_probs(
        elo_a,
        elo_b,
        surface_wr_a=surface_wr_a,
        surface_wr_b=surface_wr_b,
        adjustments=adjustments,
        serve_elo_a=serve_elo_a,
        return_elo_a=return_elo_a,
        serve_elo_b=serve_elo_b,
        return_elo_b=return_elo_b,
        cpi_factor=cpi_factor,
    )
    pa = pressure_a or {}
    pb = pressure_b or {}
    n_a = int(pa.get("n_effective") or pa.get("n_matches") or pa.get("n_dataset") or pa.get("n_charted") or 0)
    n_b = int(pb.get("n_effective") or pb.get("n_matches") or pb.get("n_dataset") or pb.get("n_charted") or 0)
    p_static = match_win_prob(p_serve_a, p_serve_b, best_of=best_of)
    p_dynamic = match_win_prob_pressure(
        p_serve_a,
        p_serve_b,
        best_of=best_of,
        bp_save_a=pa.get("bp_save_rate"),
        bp_save_b=pb.get("bp_save_rate"),
        tb_clutch_a=pa.get("tb_clutch"),
        tb_clutch_b=pb.get("tb_clutch"),
        n_matches_a=n_a,
        n_matches_b=n_b,
    )
    return {
        "p_serve_a": round(p_serve_a, 4),
        "p_serve_b": round(p_serve_b, 4),
        "p_markov_static": round(p_static, 4),
        "p_markov": round(0.35 * p_static + 0.65 * p_dynamic, 4),
        "pressure_a": pa,
        "pressure_b": pb,
        "mcp_shrink_a": n_a < DATASET_LOW_SAMPLE,
        "mcp_shrink_b": n_b < DATASET_LOW_SAMPLE,
    }
