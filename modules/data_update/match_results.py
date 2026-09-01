"""Risoluzione risultati match per settle: TML → Betfair → ESPN → RapidAPI → TA → UTS → SofaScore → FlashScore → tennis-data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

SETTLE_SOURCES = (
    "tml",
    "sackmann",
    "betfair_settled",
    "espn",
    "rapidapi_tennis",
    "rapidapi_sofascore",
    "tennis_abstract_charting",
    "tennis_abstract_tourney",
    "uts_final",
    "sofascore",
    "flashscore",
    "tennis-data",
)


@dataclass
class SettleResult:
    winner: str
    score: str | None
    source: str
    loser: str | None = None


@dataclass
class ResultProviders:
    """Cache provider per batch settle (evita fetch ripetuti)."""

    days: int = 14
    _tml: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _sackmann: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _betfair: list[dict] | None = field(default=None, init=False, repr=False)
    _flashscore: list[dict] | None = field(default=None, init=False, repr=False)
    _espn: list[dict] | None = field(default=None, init=False, repr=False)
    _rapidapi: list[dict] | None = field(default=None, init=False, repr=False)
    _tennis_abstract: list[dict] | None = field(default=None, init=False, repr=False)
    _uts: list[dict] | None = field(default=None, init=False, repr=False)
    _sofascore: list[dict] | None = field(default=None, init=False, repr=False)
    _tennis_data: pd.DataFrame | None = field(default=None, init=False, repr=False)
    stats: dict[str, Any] = field(default_factory=dict)

    def tml(self) -> pd.DataFrame:
        if self._tml is None:
            self._tml = _load_recent_tml(days=self.days)
            self.stats["tml_rows"] = int(len(self._tml))
        return self._tml

    def sackmann(self) -> pd.DataFrame:
        if self._sackmann is None:
            self._sackmann = _load_recent_sackmann(days=self.days)
            self.stats["sackmann_rows"] = int(len(self._sackmann))
        return self._sackmann

    def betfair(self) -> list[dict]:
        if self._betfair is None:
            from modules.data_update.betfair import load_betfair_settled_results

            self._betfair = load_betfair_settled_results(days=self.days)
            self.stats["betfair_rows"] = len(self._betfair)
        return self._betfair

    def flashscore(self) -> list[dict]:
        if self._flashscore is None:
            from modules.data_update.flashscore import load_flashscore_results

            self._flashscore = load_flashscore_results()
            self.stats["flashscore_rows"] = len(self._flashscore)
        return self._flashscore

    def espn(self) -> list[dict]:
        if self._espn is None:
            from modules.data_update.espn_livescore import load_espn_results

            self._espn = load_espn_results(days=self.days)
            self.stats["espn_rows"] = len(self._espn)
        return self._espn

    def rapidapi(self) -> list[dict]:
        if self._rapidapi is None:
            from modules.data_update.rapidapi_tennis import load_rapidapi_results

            self._rapidapi = load_rapidapi_results(days=self.days)
            self.stats["rapidapi_rows"] = len(self._rapidapi)
        return self._rapidapi

    def tennis_abstract(self) -> list[dict]:
        if self._tennis_abstract is None:
            from modules.data_update.tennis_abstract_results import load_tennis_abstract_results

            self._tennis_abstract = load_tennis_abstract_results(days=self.days)
            self.stats["tennis_abstract_rows"] = len(self._tennis_abstract)
        return self._tennis_abstract

    def uts(self) -> list[dict]:
        if self._uts is None:
            from modules.data_update.uts_results import load_uts_results

            self._uts = load_uts_results(days=max(self.days, 30))
            self.stats["uts_rows"] = len(self._uts)
        return self._uts

    def sofascore(self) -> list[dict]:
        if self._sofascore is None:
            from modules.data_update.sofascore_livescore import load_sofascore_results

            self._sofascore = load_sofascore_results(days=self.days)
            self.stats["sofascore_rows"] = len(self._sofascore)
        return self._sofascore

    def tennis_data(self) -> pd.DataFrame:
        if self._tennis_data is None:
            self._tennis_data = _load_recent_tennis_data(days=self.days)
            self.stats["tennis_data_rows"] = int(len(self._tennis_data))
        return self._tennis_data


def _load_recent_tml(*, days: int) -> pd.DataFrame:
    from modules.data_update.tml import load_tml_matches

    cutoff = datetime.now() - timedelta(days=days)
    df = load_tml_matches(min_year=cutoff.year - 1)
    if df.empty:
        return df
    df = df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    return df[df["tourney_date"] >= cutoff]


def _load_recent_sackmann(*, days: int) -> pd.DataFrame:
    from modules.data_update.sackmann import load_sackmann_matches

    cutoff = datetime.now() - timedelta(days=days)
    frames: list[pd.DataFrame] = []
    for tour in ("atp", "wta"):
        try:
            df = load_sackmann_matches(tour=tour, min_year=cutoff.year - 1)
            if not df.empty:
                frames.append(df)
        except FileNotFoundError:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["tourney_date"] = pd.to_datetime(out["tourney_date"], errors="coerce")
    return out[out["tourney_date"] >= cutoff]


def _load_recent_tennis_data(*, days: int) -> pd.DataFrame:
    from modules.data_update.tennis_data_portal import load_odds_all

    df = load_odds_all()
    if df.empty:
        return df
    date_col = next((c for c in ("Date", "date") if c in df.columns), None)
    w_col = next((c for c in ("Winner", "winner") if c in df.columns), None)
    l_col = next((c for c in ("Loser", "loser") if c in df.columns), None)
    if not date_col or not w_col or not l_col:
        return pd.DataFrame()
    out = df.copy()
    out["_date"] = pd.to_datetime(out[date_col], errors="coerce")
    cutoff = datetime.now() - timedelta(days=days)
    return out[out["_date"] >= cutoff]


def _names_match(a: str, b: str, x: str, y: str) -> bool:
    from modules.data_update.entity_resolution import player_side_match

    direct = player_side_match(a, x) and player_side_match(b, y)
    swap = player_side_match(a, y) and player_side_match(b, x)
    if direct or swap:
        return True
    # Fallback stretto per formati tennis-data (cognome + iniziale)
    from modules.data_update.entity_resolution import _last_name

    la, lb, lx, ly = _last_name(a), _last_name(b), _last_name(x), _last_name(y)
    if not all((la, lb, lx, ly)):
        return False
    return (la == lx and lb == ly) or (la == ly and lb == lx)


def infer_tour_from_context(*, tour: str | None, tourney: str | None) -> str:
    t = str(tour or "").upper()
    if t in ("ATP", "WTA"):
        return t
    name = str(tourney or "").lower()
    if any(k in name for k in ("women", " wta", "wta ", "ladies", "female")):
        return "WTA"
    if any(k in name for k in ("men", " atp", "atp ", "gentlemen", "male")):
        return "ATP"
    return "ATP"


def _date_ok(match_day: str, result_day: str | None, *, tolerance: int) -> bool:
    if not match_day or not result_day:
        return True
    try:
        return abs((pd.Timestamp(result_day) - pd.Timestamp(match_day)).days) <= tolerance
    except Exception:
        return False


def _from_dataframe(
    df: pd.DataFrame,
    *,
    player_a: str,
    player_b: str,
    day: str,
    source: str,
    tolerance: int,
) -> SettleResult | None:
    if df.empty:
        return None
    for _, row in df.iterrows():
        rday = pd.Timestamp(row.get("tourney_date") or row.get("_date")).strftime("%Y-%m-%d")
        if not _date_ok(day, rday, tolerance=tolerance):
            continue
        wname = str(row.get("winner_name") or row.get("Winner") or row.get("winner") or "")
        lname = str(row.get("loser_name") or row.get("Loser") or row.get("loser") or "")
        if not wname or not lname:
            continue
        if not _names_match(player_a, player_b, wname, lname):
            continue
        score_col = next((c for c in ("score", "Score", "SC") if c in row.index), None)
        score = str(row[score_col]) if score_col and pd.notna(row.get(score_col)) else None
        return SettleResult(winner=wname, loser=lname, score=score, source=source)
    return None


def _from_betfair(
    rows: list[dict],
    *,
    player_a: str,
    player_b: str,
    day: str,
    tolerance: int,
) -> SettleResult | None:
    for row in rows:
        pa = str(row.get("player_a") or "")
        pb = str(row.get("player_b") or "")
        if not _names_match(player_a, player_b, pa, pb):
            continue
        rday = str(row.get("date") or row.get("commence_time") or "")[:10]
        if not _date_ok(day, rday, tolerance=tolerance):
            continue
        winner = str(row.get("winner") or "")
        if not winner:
            continue
        return SettleResult(
            winner=winner,
            loser=str(row.get("loser") or ""),
            score=row.get("score"),
            source="betfair_settled",
        )
    return None


def _from_live_rows(
    rows: list[dict],
    *,
    player_a: str,
    player_b: str,
    day: str,
    source: str,
    tolerance: int,
    tour: str | None = None,
) -> SettleResult | None:
    want_tour = str(tour or "").upper() if tour else ""
    for row in rows:
        if want_tour:
            row_tour = str(row.get("tour") or "").upper()
            if row_tour and row_tour != want_tour:
                continue
        pa = str(row.get("player_a") or "")
        pb = str(row.get("player_b") or "")
        if not _names_match(player_a, player_b, pa, pb):
            continue
        rday = str(row.get("date") or "")[:10]
        if rday and not _date_ok(day, rday, tolerance=tolerance):
            continue
        winner = str(row.get("winner") or "")
        if not winner:
            continue
        row_source = str(row.get("source") or source)
        return SettleResult(
            winner=winner,
            loser=str(row.get("loser") or ""),
            score=row.get("score"),
            source=row_source,
        )
    return None


def _from_flashscore(
    rows: list[dict],
    *,
    player_a: str,
    player_b: str,
    day: str,
    tolerance: int,
    tour: str | None = None,
) -> SettleResult | None:
    return _from_live_rows(
        rows,
        player_a=player_a,
        player_b=player_b,
        day=day,
        source="flashscore",
        tolerance=tolerance,
        tour=tour,
    )


def resolve_match_result(
    player_a: str,
    player_b: str,
    *,
    date: str,
    tour: str = "ATP",
    providers: ResultProviders | None = None,
    day_tolerance: int = 2,
    tourney: str | None = None,
) -> SettleResult | None:
    """Cascade: TML → Sackmann → Betfair → ESPN → RapidAPI → TA → UTS → SofaScore → FlashScore → tennis-data."""
    prov = providers or ResultProviders()
    day = str(date or "")[:10]
    tour = infer_tour_from_context(tour=tour, tourney=tourney)

    # A — TML (ATP primario)
    if tour == "ATP":
        hit = _from_dataframe(
            prov.tml(), player_a=player_a, player_b=player_b, day=day, source="tml", tolerance=day_tolerance
        )
        if hit:
            return hit

    # A — Sackmann (WTA o gap-fill ATP)
    hit = _from_dataframe(
        prov.sackmann(),
        player_a=player_a,
        player_b=player_b,
        day=day,
        source="sackmann",
        tolerance=day_tolerance,
    )
    if hit:
        return hit

    # B — Betfair mercati settled
    hit = _from_betfair(
        prov.betfair(), player_a=player_a, player_b=player_b, day=day, tolerance=day_tolerance
    )
    if hit:
        return hit

    # C — ESPN live scoreboard (Slam + tornei recenti)
    hit = _from_live_rows(
        prov.espn(),
        player_a=player_a,
        player_b=player_b,
        day=day,
        source="espn",
        tolerance=day_tolerance,
        tour=tour,
    )
    if hit:
        return hit
    if tour:
        hit = _from_live_rows(
            prov.espn(),
            player_a=player_a,
            player_b=player_b,
            day=day,
            source="espn",
            tolerance=day_tolerance,
            tour=None,
        )
        if hit:
            return hit

    # D — RapidAPI (SofaScore wrapper + opzionale Tennis API)
    hit = _from_live_rows(
        prov.rapidapi(),
        player_a=player_a,
        player_b=player_b,
        day=day,
        source="rapidapi_sofascore",
        tolerance=day_tolerance,
        tour=tour,
    )
    if hit:
        return hit
    if tour:
        hit = _from_live_rows(
            prov.rapidapi(),
            player_a=player_a,
            player_b=player_b,
            day=day,
            source="rapidapi_sofascore",
            tolerance=day_tolerance,
            tour=None,
        )
        if hit:
            return hit

    # E — Tennis Abstract charting (punto-per-punto, Slam)
    hit = _from_live_rows(
        prov.tennis_abstract(),
        player_a=player_a,
        player_b=player_b,
        day=day,
        source="tennis_abstract_charting",
        tolerance=day_tolerance,
        tour=tour,
    )
    if hit:
        return hit

    # F — UTS (finali torneo + metadati)
    hit = _from_live_rows(
        prov.uts(),
        player_a=player_a,
        player_b=player_b,
        day=day,
        source="uts_final",
        tolerance=day_tolerance,
    )
    if hit:
        return hit

    # G — SofaScore diretto (curl_cffi)
    hit = _from_live_rows(
        prov.sofascore(),
        player_a=player_a,
        player_b=player_b,
        day=day,
        source="sofascore",
        tolerance=day_tolerance,
    )
    if hit:
        return hit

    # H — FlashScore / diretta.it
    hit = _from_flashscore(
        prov.flashscore(),
        player_a=player_a,
        player_b=player_b,
        day=day,
        tolerance=day_tolerance,
        tour=tour,
    )
    if hit:
        return hit

    # I — tennis-data.co.uk (risultati + quote storiche)
    hit = _from_dataframe(
        prov.tennis_data(),
        player_a=player_a,
        player_b=player_b,
        day=day,
        source="tennis-data",
        tolerance=day_tolerance,
    )
    if hit:
        return hit

    return None
