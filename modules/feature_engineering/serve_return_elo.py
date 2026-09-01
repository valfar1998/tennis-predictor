"""Serve-Elo e Return-Elo disaggregati per superficie (Barnett & Clarke input)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from modules.constants import ELO_START, INACTIVITY_DAYS, INACTIVITY_DECAY
from modules.feature_engineering.elo import expected_score, k_factor


@dataclass
class PlayerServeReturn:
    serve_global: float = ELO_START
    return_global: float = ELO_START
    serve_surface: dict[str, float] = field(default_factory=dict)
    return_surface: dict[str, float] = field(default_factory=dict)
    last_match: datetime | None = None
    n_matches: int = 0

    def serve_blended(self, surface: str, weight: float = 0.68) -> float:
        rs = self.serve_surface.get(surface, self.serve_global)
        return weight * rs + (1.0 - weight) * self.serve_global

    def return_blended(self, surface: str, weight: float = 0.68) -> float:
        rr = self.return_surface.get(surface, self.return_global)
        return weight * rr + (1.0 - weight) * self.return_global


def _point_serve_expected(serve_r: float, return_r: float, *, base: float = 0.62, scale: float = 0.0008) -> float:
    return max(0.45, min(0.78, base + scale * (serve_r - return_r)))


def _serve_stats(row: pd.Series, prefix: str) -> float | None:
    svpt = row.get(f"{prefix}_svpt")
    if pd.isna(svpt) or float(svpt) <= 0:
        return None
    w1 = float(row.get(f"{prefix}_1stWon") or 0)
    w2 = float(row.get(f"{prefix}_2ndWon") or 0)
    return (w1 + w2) / float(svpt)


class ServeReturnEloEngine:
    """Elo separato servizio/risposta aggiornato da statistiche punto."""

    def __init__(self, *, surface_weight: float = 0.68):
        self.players: dict[int, PlayerServeReturn] = {}
        self.surface_weight = surface_weight

    def _get(self, pid: int) -> PlayerServeReturn:
        if pid not in self.players:
            pe = PlayerServeReturn()
            for s in ("Hard", "Clay", "Grass", "Carpet"):
                pe.serve_surface[s] = ELO_START
                pe.return_surface[s] = ELO_START
            self.players[pid] = pe
        return self.players[pid]

    def _apply_inactivity(self, pe: PlayerServeReturn, match_date: datetime) -> None:
        if pe.last_match is None:
            return
        if (match_date - pe.last_match).days > INACTIVITY_DAYS:
            pe.serve_global = ELO_START + (pe.serve_global - ELO_START) * INACTIVITY_DECAY
            pe.return_global = ELO_START + (pe.return_global - ELO_START) * INACTIVITY_DECAY
            for s in pe.serve_surface:
                pe.serve_surface[s] = ELO_START + (pe.serve_surface[s] - ELO_START) * INACTIVITY_DECAY
                pe.return_surface[s] = ELO_START + (pe.return_surface[s] - ELO_START) * INACTIVITY_DECAY

    def pre_match_ratings(
        self,
        player_a: int,
        player_b: int,
        surface: str,
        match_date: datetime,
    ) -> tuple[float, float, float, float]:
        pa, pb = self._get(player_a), self._get(player_b)
        self._apply_inactivity(pa, match_date)
        self._apply_inactivity(pb, match_date)
        w = self.surface_weight
        return (
            pa.serve_blended(surface, w),
            pa.return_blended(surface, w),
            pb.serve_blended(surface, w),
            pb.return_blended(surface, w),
        )

    def update_from_match(
        self,
        winner_id: int,
        loser_id: int,
        row: pd.Series,
        *,
        surface: str,
        level: str | None,
        match_date: datetime,
    ) -> None:
        pw, pl = self._get(winner_id), self._get(loser_id)
        self._apply_inactivity(pw, match_date)
        self._apply_inactivity(pl, match_date)

        k = k_factor(level) * 0.85
        w = self.surface_weight

        serve_w_pre = pw.serve_blended(surface, w)
        ret_w_pre = pw.return_blended(surface, w)
        serve_l_pre = pl.serve_blended(surface, w)
        ret_l_pre = pl.return_blended(surface, w)

        p_serve_w = _serve_stats(row, "w")
        p_serve_l = _serve_stats(row, "l")

        if p_serve_w is not None:
            exp_w = _point_serve_expected(serve_w_pre, ret_l_pre)
            delta = k * (p_serve_w - exp_w)
            pw.serve_global += delta
            pw.serve_surface[surface] = pw.serve_surface.get(surface, ELO_START) + delta
            pl.return_global -= delta * 0.45
            pl.return_surface[surface] = pl.return_surface.get(surface, ELO_START) - delta * 0.45

        if p_serve_l is not None:
            exp_l = _point_serve_expected(serve_l_pre, ret_w_pre)
            delta_l = k * (p_serve_l - exp_l)
            pl.serve_global += delta_l
            pl.serve_surface[surface] = pl.serve_surface.get(surface, ELO_START) + delta_l
            pw.return_global -= delta_l * 0.45
            pw.return_surface[surface] = pw.return_surface.get(surface, ELO_START) - delta_l * 0.45

        if p_serve_w is None and p_serve_l is None:
            # Fallback: esito match sposta serve/return in modo asimmetrico
            e_match = expected_score(serve_w_pre - ret_l_pre * 0.3, serve_l_pre - ret_w_pre * 0.3)
            delta_m = k_factor(level) * (1.0 - e_match)
            pw.serve_global += delta_m * 0.6
            pw.return_global += delta_m * 0.4
            pl.serve_global -= delta_m * 0.6
            pl.return_global -= delta_m * 0.4
            pw.serve_surface[surface] = pw.serve_surface.get(surface, ELO_START) + delta_m * 0.6
            pl.serve_surface[surface] = pl.serve_surface.get(surface, ELO_START) - delta_m * 0.6

        pw.last_match = pl.last_match = match_date
        pw.n_matches += 1
        pl.n_matches += 1

    def run_chronological(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        m = matches.sort_values("tourney_date")
        for _, row in m.iterrows():
            wid, lid = int(row["winner_id"]), int(row["loser_id"])
            surface = str(row.get("surface_norm") or row.get("surface") or "Hard")
            dt = pd.Timestamp(row["tourney_date"]).to_pydatetime()
            sw, rw, sl, rl = self.pre_match_ratings(wid, lid, surface, dt)
            rows.append({
                "tourney_date": dt,
                "winner_id": wid,
                "loser_id": lid,
                "serve_elo_w": sw,
                "return_elo_w": rw,
                "serve_elo_l": sl,
                "return_elo_l": rl,
                "serve_return_diff_a": sw - rl,
            })
            self.update_from_match(
                wid, lid, row, surface=surface, level=row.get("tourney_level"), match_date=dt
            )
        return pd.DataFrame(rows)
