"""Walk-forward validation con soglie Fase 1/2 (anti-overfit / drift superficie-Elo).

Non modifica MIN_EDGE/odds/Kelly: riproduce la policy corrente su finestre OOF
temporali e confronta ROI/hit per superficie e stabilità complessiva.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from modules.advisor.staking import fractional_kelly
from modules.advisor.value import devig_shin
from modules.constants import (
    MAX_ODDS_PLAY,
    MIN_EDGE,
    MIN_KELLY,
    MIN_ODDS_PLAY,
    MIN_PROB_PLAY,
    PHASE2_EDGE_FLOOR,
)

ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "data" / "models" / "oof_predictions.joblib"
REPORT_PATH = ROOT / "data" / "processed" / "walk_forward_report.json"


def _max_drawdown(path: list[float]) -> float:
    peak = path[0] if path else 1.0
    max_dd = 0.0
    for v in path:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)
    return max_dd


def _simulate_window(
    df: pd.DataFrame,
    *,
    min_edge: float,
    min_prob: float = MIN_PROB_PLAY,
) -> dict[str, Any]:
    bets: list[dict[str, Any]] = []
    bankroll = 1.0
    path = [1.0]

    for _, row in df.iterrows():
        p_a = float(row.get("p_ml") or row.get("p_elo") or 0.5)
        p_b = 1.0 - p_a
        try:
            ow, ol = float(row["odds_winner"]), float(row["odds_loser"])
        except (TypeError, ValueError, KeyError):
            continue
        if ow <= 1.01 or ol <= 1.01:
            continue
        # Side A = winner storico nel OOF (p_ml ≈ P(winner))
        for side, p, odds in (("A", p_a, ow), ("B", p_b, ol)):
            if odds < MIN_ODDS_PLAY or odds > MAX_ODDS_PLAY:
                continue
            ev = p * odds - 1.0
            if ev < min_edge or p < min_prob:
                continue
            stake = fractional_kelly(p, odds)
            if stake < MIN_KELLY:
                continue
            hit = 1 if side == "A" else 0
            profit = stake * (odds - 1.0) if hit else -stake
            bankroll += profit
            path.append(bankroll)
            bets.append(
                {
                    "surface": str(row.get("surface") or "Unknown"),
                    "stake": stake,
                    "profit": profit,
                    "hit": hit,
                    "ev": ev,
                    "p_elo": float(row.get("p_elo") or 0) if pd.notna(row.get("p_elo")) else None,
                    "p_ml": float(row.get("p_ml") or 0) if pd.notna(row.get("p_ml")) else None,
                }
            )

    if not bets:
        return {
            "n_bets": 0,
            "roi": 0.0,
            "hit_rate": None,
            "max_drawdown": 0.0,
            "final_bankroll": 1.0,
            "by_surface": [],
            "elo_ml_gap_mean": None,
        }

    bdf = pd.DataFrame(bets)
    total_staked = float(bdf["stake"].sum())
    roi = float(bdf["profit"].sum() / total_staked) if total_staked > 0 else 0.0
    by_surface = (
        bdf.groupby("surface")
        .agg(
            n=("hit", "count"),
            hit_rate=("hit", "mean"),
            stake=("stake", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    by_surface["roi"] = by_surface.apply(
        lambda r: (r["profit"] / r["stake"]) if r["stake"] > 0 else 0.0, axis=1
    )
    gaps = []
    for b in bets:
        if b.get("p_elo") is not None and b.get("p_ml") is not None:
            gaps.append(abs(float(b["p_ml"]) - float(b["p_elo"])))

    return {
        "n_bets": int(len(bdf)),
        "roi": round(roi, 4),
        "hit_rate": round(float(bdf["hit"].mean()), 4),
        "max_drawdown": round(_max_drawdown(path), 4),
        "final_bankroll": round(bankroll, 4),
        "mean_ev": round(float(bdf["ev"].mean()), 4),
        "by_surface": by_surface[["surface", "n", "hit_rate", "roi"]].to_dict("records"),
        "elo_ml_gap_mean": round(sum(gaps) / len(gaps), 4) if gaps else None,
    }


def _drift_flags(windows: list[dict[str, Any]]) -> list[str]:
    """Segnali soft di overfitting / drift (non bloccanti)."""
    flags: list[str] = []
    rois = [w["metrics"]["roi"] for w in windows if w["metrics"]["n_bets"] >= 20]
    if len(rois) >= 3:
        early = sum(rois[: len(rois) // 2]) / max(1, len(rois) // 2)
        late = sum(rois[len(rois) // 2 :]) / max(1, len(rois) - len(rois) // 2)
        if early - late > 0.08:
            flags.append(f"roi_decay: early {early:.1%} → late {late:.1%}")
        if max(rois) - min(rois) > 0.15:
            flags.append(f"roi_unstable: spread {max(rois) - min(rois):.1%}")

    # Surface drift: confronta ROI Hard vs Clay sull'ultima finestra piena
    for w in reversed(windows):
        surf = {r["surface"]: r for r in w["metrics"].get("by_surface") or []}
        hard = surf.get("Hard") or surf.get("hard")
        clay = surf.get("Clay") or surf.get("clay")
        if hard and clay and hard.get("n", 0) >= 15 and clay.get("n", 0) >= 15:
            gap = abs(float(hard.get("roi") or 0) - float(clay.get("roi") or 0))
            if gap > 0.12:
                flags.append(
                    f"surface_gap {w['label']}: Hard {hard.get('roi'):.1%} vs Clay {clay.get('roi'):.1%}"
                )
            break

    gaps = [
        w["metrics"]["elo_ml_gap_mean"]
        for w in windows
        if w["metrics"].get("elo_ml_gap_mean") is not None
    ]
    if len(gaps) >= 3 and gaps[-1] - gaps[0] > 0.04:
        flags.append(f"elo_ml_divergence: {gaps[0]:.3f} → {gaps[-1]:.3f}")

    return flags


def run_walk_forward(
    oof: pd.DataFrame | None = None,
    *,
    n_windows: int = 5,
    min_edge: float | None = None,
    also_phase2_floor: bool = True,
) -> dict[str, Any]:
    """Walk-forward temporale su OOF con policy Fase 1 (e opzionale floor Fase 2)."""
    if oof is None:
        if not OOF_PATH.exists():
            return {"ok": False, "error": "OOF non trovato — esegui prima train/retrain"}
        oof = joblib.load(OOF_PATH)

    df = oof.dropna(subset=["odds_winner", "odds_loser"]).copy()
    if "tourney_date" not in df.columns:
        return {"ok": False, "error": "OOF senza tourney_date"}
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"]).sort_values("tourney_date")
    if df.empty:
        return {"ok": False, "error": "Nessun match OOF con quote/date"}

    edge = float(min_edge if min_edge is not None else MIN_EDGE)
    dates = df["tourney_date"].sort_values().unique()
    n_windows = max(2, min(int(n_windows), 12))
    # Finestre consecutive non-overlapping (test folds)
    chunk = max(1, len(dates) // n_windows)
    windows: list[dict[str, Any]] = []

    for i in range(n_windows):
        start_i = i * chunk
        end_i = (i + 1) * chunk if i < n_windows - 1 else len(dates)
        if start_i >= len(dates):
            break
        d0, d1 = dates[start_i], dates[end_i - 1]
        fold = df[(df["tourney_date"] >= d0) & (df["tourney_date"] <= d1)]
        label = f"{pd.Timestamp(d0).date()}→{pd.Timestamp(d1).date()}"
        metrics = _simulate_window(fold, min_edge=edge)
        entry: dict[str, Any] = {
            "label": label,
            "n_matches": int(len(fold)),
            "min_edge": edge,
            "metrics": metrics,
        }
        if also_phase2_floor and abs(edge - PHASE2_EDGE_FLOOR) > 1e-9:
            entry["metrics_phase2_floor"] = _simulate_window(fold, min_edge=PHASE2_EDGE_FLOOR)
        windows.append(entry)

    flags = _drift_flags(windows)
    # Aggregate full-sample sanity under current edge
    full = _simulate_window(df, min_edge=edge)

    report: dict[str, Any] = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "min_edge": edge,
            "min_odds": MIN_ODDS_PLAY,
            "max_odds": MAX_ODDS_PLAY,
            "min_kelly": MIN_KELLY,
            "min_prob": MIN_PROB_PLAY,
            "phase2_floor": PHASE2_EDGE_FLOOR,
        },
        "n_windows": len(windows),
        "windows": windows,
        "full_sample": full,
        "drift_flags": flags,
        "healthy": len(flags) == 0,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def format_walk_forward_banner(report: dict[str, Any] | None = None) -> str:
    if report is None:
        if REPORT_PATH.is_file():
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        else:
            return "WALK-FORWARD · no report"
    if not report.get("ok"):
        return f"WALK-FORWARD · ERR {report.get('error')}"
    full = report.get("full_sample") or {}
    flags = report.get("drift_flags") or []
    status = "OK" if report.get("healthy") else f"FLAGS={len(flags)}"
    return (
        f"WALK-FORWARD · {status} windows={report.get('n_windows')} "
        f"full_n={full.get('n_bets')} ROI={full.get('roi')} "
        f"hit={full.get('hit_rate')} DD={full.get('max_drawdown')} "
        f"edge={((report.get('policy') or {}).get('min_edge'))}"
    )
