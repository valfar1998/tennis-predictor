"""Backtesting engine senza data leakage su quote storiche."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.advisor.staking import beat_close, clv_prob, fractional_kelly
from modules.advisor.value import devig_shin
from modules.calibration.config import save_calibration
from modules.constants import MIN_EDGE

ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "data" / "models" / "oof_predictions.joblib"


def run_backtest(
    oof: pd.DataFrame | None = None,
    *,
    min_edge: float = MIN_EDGE,
    min_prob: float = 0.38,
) -> dict:
    """Simula value betting su OOF predictions con quote storiche."""
    if oof is None:
        if not OOF_PATH.exists():
            return {"ok": False, "error": "OOF non trovato — esegui prima il training"}
        oof = joblib.load(OOF_PATH)

    df = oof.dropna(subset=["odds_winner", "odds_loser"]).copy()
    if df.empty:
        return {"ok": False, "error": "Nessun match con quote nel dataset"}

    bets: list[dict] = []
    bankroll = 1.0
    path = [1.0]

    for _, row in df.iterrows():
        p_a = float(row.get("p_ml") or row.get("p_elo") or 0.5)
        p_b = 1 - p_a
        ow, ol = float(row["odds_winner"]), float(row["odds_loser"])
        mkt_a, mkt_b = devig_shin(ow, ol)

        for side, p, odds in [("A", p_a, ow), ("B", p_b, ol)]:
            ev = p * odds - 1
            if ev < min_edge or p < min_prob:
                continue
            stake = fractional_kelly(p, odds)
            if stake <= 0:
                continue
            hit = 1 if side == "A" else 0
            profit = stake * (odds - 1) if hit else -stake
            bankroll += profit
            path.append(bankroll)
            bets.append({
                "date": row.get("tourney_date"),
                "side": side,
                "prob": p,
                "odds": odds,
                "ev": ev,
                "stake": stake,
                "hit": hit,
                "profit": profit,
                "surface": row.get("surface"),
                "clv": clv_prob(odds, odds),
                "note": "CLV richiede quote open vs Pinnacle close distinte",
            })

    if not bets:
        return {"ok": True, "n_bets": 0, "roi": 0, "hit_rate": 0}

    bdf = pd.DataFrame(bets)
    total_staked = bdf["stake"].sum()
    roi = bdf["profit"].sum() / total_staked if total_staked > 0 else 0
    hit_rate = bdf["hit"].mean()
    mean_ev = bdf["ev"].mean()
    realization = roi / mean_ev if mean_ev > 0 else 0

    by_surface = (
        bdf.groupby("surface")
        .agg(n=("hit", "count"), roi=("profit", lambda x: x.sum() / bdf.loc[x.index, "stake"].sum()), hit_rate=("hit", "mean"))
        .reset_index()
        .to_dict("records")
    )

    summary = {
        "ok": True,
        "n_bets": len(bdf),
        "roi": round(roi, 4),
        "hit_rate": round(hit_rate, 4),
        "mean_ev": round(mean_ev, 4),
        "realization": round(realization, 4),
        "final_bankroll": round(bankroll, 4),
        "max_drawdown": round(_max_drawdown(path), 4),
        "by_surface": by_surface,
    }

    cal = {"backtest_summary": summary, "by_surface": by_surface, "min_ev_play": min_edge}
    save_calibration(cal)
    return summary


def _max_drawdown(path: list[float]) -> float:
    peak = path[0]
    max_dd = 0.0
    for v in path:
        peak = max(peak, v)
        dd = (v - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)
    return max_dd
