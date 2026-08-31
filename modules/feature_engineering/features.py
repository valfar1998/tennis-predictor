"""Feature engineering: fatica, H2H, ranking, form superficie."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from modules.feature_engineering.elo import EloEngine, expected_score
from modules.dataset_loader.loader import load_matches

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

FEATURE_COLS = [
    "elo_diff",
    "e_w_elo",
    "rank_diff",
    "rank_points_diff",
    "h2h_wins_a",
    "h2h_surface_wins_a",
    "fatigue_minutes_7d_a",
    "fatigue_minutes_7d_b",
    "rest_days_a",
    "rest_days_b",
    "surface_wr_a",
    "surface_wr_b",
    "form_wr_10_a",
    "form_wr_10_b",
    "level_weight",
    "best_of",
    "is_bo5",
    "mkt_p_a",
    "mkt_overround",
]


def _flip_perspective(feat: dict) -> dict:
    """Riga dal punto di vista del perdente (player A = loser)."""
    flipped = dict(feat)
    flipped["winner_id"], flipped["loser_id"] = feat["loser_id"], feat["winner_id"]
    flipped["winner_name"], flipped["loser_name"] = feat["loser_name"], feat["winner_name"]
    flipped["elo_w_pre"], flipped["elo_l_pre"] = feat["elo_l_pre"], feat["elo_w_pre"]
    flipped["elo_diff"] = -feat["elo_diff"]
    flipped["e_w_elo"] = 1.0 - feat["e_w_elo"]
    flipped["rank_diff"] = -feat["rank_diff"]
    flipped["rank_points_diff"] = -feat["rank_points_diff"]
    flipped["h2h_wins_a"] = 0  # perspective is loser
    flipped["fatigue_minutes_7d_a"], flipped["fatigue_minutes_7d_b"] = (
        feat["fatigue_minutes_7d_b"], feat["fatigue_minutes_7d_a"]
    )
    flipped["rest_days_a"], flipped["rest_days_b"] = feat["rest_days_b"], feat["rest_days_a"]
    flipped["surface_wr_a"], flipped["surface_wr_b"] = feat["surface_wr_b"], feat["surface_wr_a"]
    flipped["form_wr_10_a"], flipped["form_wr_10_b"] = feat["form_wr_10_b"], feat["form_wr_10_a"]
    if feat.get("mkt_p_a") is not None:
        flipped["mkt_p_a"] = 1.0 - feat["mkt_p_a"]
    flipped["odds_winner"], flipped["odds_loser"] = feat.get("odds_loser"), feat.get("odds_winner")
    flipped["label"] = 0
    return flipped


class FeatureEngineer:
    def __init__(self):
        self.h2h: dict[tuple, list] = defaultdict(list)
        self.recent: dict[int, list] = defaultdict(list)
        self.surface_form: dict[tuple, list] = defaultdict(list)

    def _pair_key(self, a: int, b: int) -> tuple:
        return (min(a, b), max(a, b))

    def _fatigue(self, pid: int, dt, window_days: int = 7) -> float:
        cutoff = dt - timedelta(days=window_days)
        mins = [m for d, m in self.recent.get(pid, []) if d >= cutoff]
        return float(sum(mins))

    def _rest_days(self, pid: int, dt) -> float:
        hist = self.recent.get(pid, [])
        if not hist:
            return 14.0
        last = max(d for d, _ in hist)
        return max(0.0, (dt - last).days)

    def _surface_wr(self, pid: int, surface: str, n: int = 15) -> float:
        key = (pid, surface)
        wins = self.surface_form.get(key, [])
        if not wins:
            return 0.5
        recent = wins[-n:]
        return sum(recent) / len(recent)

    def _form_wr(self, pid: int, n: int = 10) -> float:
        all_wins = []
        for (p, s), wins in self.surface_form.items():
            if p == pid:
                all_wins.extend(wins)
        if not all_wins:
            return 0.5
        recent = all_wins[-n:]
        return sum(recent) / len(recent)

    def _h2h_stats(self, a: int, b: int, surface: str, perspective: int) -> tuple[int, int]:
        key = self._pair_key(a, b)
        matches = self.h2h.get(key, [])
        wins_a = sum(1 for w, s in matches if w == perspective)
        surf_wins = sum(1 for w, s in matches if w == perspective and s == surface)
        return wins_a, surf_wins

    def _update_state(self, winner_id: int, loser_id: int, surface: str, dt, minutes: float):
        key = self._pair_key(winner_id, loser_id)
        self.h2h[key].append((winner_id, surface))
        for pid, won in [(winner_id, 1), (loser_id, 0)]:
            self.recent[pid].append((dt, minutes))
            self.surface_form[(pid, surface)].append(won)

    def build(self, matches: pd.DataFrame | None = None, *, save: bool = True) -> pd.DataFrame:
        matches = matches if matches is not None else load_matches()
        matches = matches.sort_values("tourney_date").reset_index(drop=True)

        elo_engine = EloEngine()
        rows: list[dict] = []

        for _, row in matches.iterrows():
            wid, lid = int(row["winner_id"]), int(row["loser_id"])
            surface = str(row.get("surface_norm") or "Hard")
            dt = pd.Timestamp(row["tourney_date"]).to_pydatetime()
            level = str(row.get("tourney_level") or "A")
            minutes = float(row.get("minutes") or 90)
            best_of = int(row.get("best_of") or 3)

            r_w, r_l = elo_engine.pre_match_ratings(wid, lid, surface, dt)
            h2h_w, h2h_surf = self._h2h_stats(wid, lid, surface, wid)

            wr = row.get("winner_rank")
            lr = row.get("loser_rank")
            rank_diff = (float(lr) - float(wr)) if pd.notna(wr) and pd.notna(lr) else 0.0
            wpt = row.get("winner_rank_points")
            lpt = row.get("loser_rank_points")
            pts_diff = (float(wpt) - float(lpt)) if pd.notna(wpt) and pd.notna(lpt) else 0.0

            ow, ol = row.get("odds_winner"), row.get("odds_loser")
            mkt_p_a, mkt_or = None, None
            if pd.notna(ow) and pd.notna(ol) and ow > 1 and ol > 1:
                imp_w, imp_l = 1 / ow, 1 / ol
                total = imp_w + imp_l
                mkt_p_a = imp_w / total
                mkt_or = total

            level_w = {"G": 1.0, "M": 0.85, "A": 0.7, "C": 0.5}.get(level, 0.65)

            feat = {
                "tourney_date": dt,
                "winner_id": wid,
                "loser_id": lid,
                "winner_name": row.get("winner_name"),
                "loser_name": row.get("loser_name"),
                "surface": surface,
                "tourney_level": level,
                "best_of": best_of,
                "is_bo5": int(best_of >= 5),
                "elo_w_pre": r_w,
                "elo_l_pre": r_l,
                "elo_diff": r_w - r_l,
                "e_w_elo": expected_score(r_w, r_l),
                "rank_diff": rank_diff,
                "rank_points_diff": pts_diff,
                "h2h_wins_a": h2h_w,
                "h2h_surface_wins_a": h2h_surf,
                "fatigue_minutes_7d_a": self._fatigue(wid, dt),
                "fatigue_minutes_7d_b": self._fatigue(lid, dt),
                "rest_days_a": self._rest_days(wid, dt),
                "rest_days_b": self._rest_days(lid, dt),
                "surface_wr_a": self._surface_wr(wid, surface),
                "surface_wr_b": self._surface_wr(lid, surface),
                "form_wr_10_a": self._form_wr(wid),
                "form_wr_10_b": self._form_wr(lid),
                "level_weight": level_w,
                "mkt_p_a": mkt_p_a,
                "mkt_overround": mkt_or,
                "odds_winner": ow,
                "odds_loser": ol,
                "label": 1,
            }
            rows.append(feat)
            rows.append(_flip_perspective(feat))

            elo_engine.update(wid, lid, surface=surface, level=level, match_date=dt)
            self._update_state(wid, lid, surface, dt, minutes)

        out = pd.DataFrame(rows)
        if save:
            FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(FEATURES_PATH, index=False)
        return out
