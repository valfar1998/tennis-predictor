"""Elo rating multisuperficie con time decay e K-factor adattivo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from modules.constants import (
    DEFAULT_K,
    ELO_START,
    ELO_SURFACE_WEIGHT,
    INACTIVITY_DAYS,
    INACTIVITY_DECAY,
    K_BY_LEVEL,
)


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def k_factor(level: str | None) -> float:
    return float(K_BY_LEVEL.get(str(level or "").upper(), DEFAULT_K))


@dataclass
class PlayerElo:
    global_rating: float = ELO_START
    surface: dict[str, float] = field(default_factory=dict)
    last_match: datetime | None = None
    n_matches: int = 0

    def blended(self, surface: str, weight: float = ELO_SURFACE_WEIGHT) -> float:
        rs = self.surface.get(surface, self.global_rating)
        return weight * rs + (1.0 - weight) * self.global_rating

    def blended_with_cpi(
        self,
        surface: str,
        cpi_norm: float | None = None,
        *,
        base_weight: float | None = None,
    ) -> float:
        """Elo con peso superficie modulato dal CPI torneo (campo rapido → più peso surface)."""
        w = base_weight if base_weight is not None else self.surface_weight
        if cpi_norm is not None:
            # cpi_norm ~0.85 slow … 1.15 fast (da cpi.py)
            adj = (float(cpi_norm) - 1.0) * 0.25
            w = max(0.52, min(0.82, w + adj))
        return self.blended(surface, w)


class EloEngine:
    def __init__(self, *, surface_weight: float = ELO_SURFACE_WEIGHT):
        self.players: dict[int, PlayerElo] = {}
        self.surface_weight = surface_weight
        self.history: list[dict] = []

    def _get(self, pid: int) -> PlayerElo:
        if pid not in self.players:
            self.players[pid] = PlayerElo()
            for s in ("Hard", "Clay", "Grass", "Carpet"):
                self.players[pid].surface[s] = ELO_START
        return self.players[pid]

    def _apply_inactivity(self, pe: PlayerElo, match_date: datetime) -> None:
        if pe.last_match is None:
            return
        days = (match_date - pe.last_match).days
        if days > INACTIVITY_DAYS:
            pe.global_rating = ELO_START + (pe.global_rating - ELO_START) * INACTIVITY_DECAY
            for s in pe.surface:
                pe.surface[s] = ELO_START + (pe.surface[s] - ELO_START) * INACTIVITY_DECAY

    def pre_match_ratings(
        self, winner_id: int, loser_id: int, surface: str, match_date: datetime
    ) -> tuple[float, float]:
        pw, pl = self._get(winner_id), self._get(loser_id)
        self._apply_inactivity(pw, match_date)
        self._apply_inactivity(pl, match_date)
        return pw.blended(surface, self.surface_weight), pl.blended(surface, self.surface_weight)

    def update(
        self,
        winner_id: int,
        loser_id: int,
        *,
        surface: str,
        level: str | None,
        match_date: datetime,
    ) -> dict:
        pw, pl = self._get(winner_id), self._get(loser_id)
        self._apply_inactivity(pw, match_date)
        self._apply_inactivity(pl, match_date)

        r_w = pw.blended(surface, self.surface_weight)
        r_l = pl.blended(surface, self.surface_weight)
        e_w = expected_score(r_w, r_l)
        k = k_factor(level)

        delta_w = k * (1.0 - e_w)
        delta_l = k * (0.0 - (1.0 - e_w))

        pw.global_rating += delta_w
        pl.global_rating += delta_l
        pw.surface[surface] = pw.surface.get(surface, ELO_START) + delta_w
        pl.surface[surface] = pl.surface.get(surface, ELO_START) + delta_l
        pw.last_match = pl.last_match = match_date
        pw.n_matches += 1
        pl.n_matches += 1

        row = {
            "date": match_date,
            "winner_id": winner_id,
            "loser_id": loser_id,
            "surface": surface,
            "r_w_pre": r_w,
            "r_l_pre": r_l,
            "e_w": e_w,
            "delta_w": delta_w,
        }
        self.history.append(row)
        return row

    def run_chronological(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Processa tutti i match in ordine cronologico, restituisce feature pre-match."""
        rows: list[dict] = []
        m = matches.sort_values("tourney_date")
        for _, row in m.iterrows():
            wid = int(row["winner_id"])
            lid = int(row["loser_id"])
            surface = str(row.get("surface_norm") or row.get("surface") or "Hard")
            dt = pd.Timestamp(row["tourney_date"]).to_pydatetime()
            level = row.get("tourney_level")

            r_w, r_l = self.pre_match_ratings(wid, lid, surface, dt)
            rows.append({
                "tourney_date": dt,
                "winner_id": wid,
                "loser_id": lid,
                "winner_name": row.get("winner_name"),
                "loser_name": row.get("loser_name"),
                "surface": surface,
                "elo_w_pre": r_w,
                "elo_l_pre": r_l,
                "elo_diff": r_w - r_l,
                "e_w_elo": expected_score(r_w, r_l),
                "tourney_level": level,
                "best_of": row.get("best_of", 3),
            })
            self.update(wid, lid, surface=surface, level=level, match_date=dt)
        return pd.DataFrame(rows)
