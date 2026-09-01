"""Dataset loader: unifica match Sackmann + quote tennis-data + cleaning."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from modules.constants import SURFACE_ALIASES
from modules.data_update.entity_resolution import vector_odds_match_keys
from modules.data_update.sackmann import load_tour_matches
from modules.data_update.tennis_data_odds import load_odds_history

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MATCHES_PATH = PROCESSED / "matches.csv"


def _normalize_surface(s: str | None) -> str:
    if not s or pd.isna(s):
        return "Hard"
    key = str(s).strip().lower()
    return SURFACE_ALIASES.get(key, str(s).strip().title())


def _is_incomplete(score: str | None) -> tuple[bool, str | None]:
    if not score or pd.isna(score):
        return True, "missing_score"
    s = str(score).upper()
    for flag in ("W/O", "WO", "WALKOVER"):
        if flag in s:
            return True, "walkover"
    for flag in ("RET", "DEF", "ABD"):
        if flag in s:
            return True, "retirement"
    return False, None


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra walkover/retirement e normalizza superfici."""
    out = df.copy()
    out["surface_norm"] = out["surface"].map(_normalize_surface)
    incomplete = out["score"].apply(lambda x: _is_incomplete(x))
    out["is_incomplete"] = incomplete.apply(lambda t: t[0])
    out["incomplete_reason"] = incomplete.apply(lambda t: t[1])
    out = out[~out["is_incomplete"]].copy()

    out["player_a_id"] = out["winner_id"]
    out["player_b_id"] = out["loser_id"]
    out["player_a_name"] = out["winner_name"]
    out["player_b_name"] = out["loser_name"]
    out["player_a_won"] = 1
    out["best_of"] = pd.to_numeric(out.get("best_of"), errors="coerce").fillna(3).astype(int)
    return out


def _serve_win_pct(w_svpt, w_won) -> float | None:
    if pd.isna(w_svpt) or w_svpt <= 0:
        return None
    return float(w_won) / float(w_svpt)


def _safe_odds_key(dt, winner: str, loser: str) -> str | None:
    """Chiave join odds; None se data invalida (NaT)."""
    keys = vector_odds_match_keys(
        pd.Series([dt]),
        pd.Series([winner]),
        pd.Series([loser]),
    )
    val = keys.iloc[0]
    return None if pd.isna(val) else str(val)


def _parse_odds_dates(series: pd.Series) -> pd.Series:
    """Parse date odds tennis-data (ISO YYYY-MM-DD o DD/MM/YYYY)."""
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    raw = series.astype(str).str.strip()
    missing = series.isna() | raw.isin(("", "nan", "none", "nat"))
    iso = ~missing & raw.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    slash = ~missing & ~iso & raw.str.contains("/", regex=False, na=False)
    other = ~missing & ~iso & ~slash
    if iso.any():
        out.loc[iso] = pd.to_datetime(series.loc[iso], errors="coerce", format="%Y-%m-%d")
    if slash.any():
        out.loc[slash] = pd.to_datetime(series.loc[slash], errors="coerce", dayfirst=True)
    if other.any():
        out.loc[other] = pd.to_datetime(series.loc[other], errors="coerce")
    return out


def add_serve_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola P(serve win) osservata per winner e loser."""
    out = df.copy()
    for prefix, w_cols, l_cols in [
        ("a", ("w_svpt", "w_1stWon", "w_2ndWon"), ("l_svpt", "l_1stWon", "l_2ndWon")),
    ]:
        svpt, w1, w2 = w_cols
        if svpt not in out.columns:
            continue
        won = out[w1].fillna(0) + out[w2].fillna(0)
        out[f"p_serve_{prefix}"] = won / out[svpt].replace(0, pd.NA)
    # Loser serve stats from l_ columns
    if "l_svpt" in out.columns:
        won_l = out["l_1stWon"].fillna(0) + out["l_2ndWon"].fillna(0)
        out["p_serve_b"] = won_l / out["l_svpt"].replace(0, pd.NA)
    return out


class DatasetLoader:
    def __init__(self, *, min_year: int = 2000):
        self.min_year = min_year

    def build(self, *, save: bool = True) -> pd.DataFrame:
        raw = load_tour_matches(min_year=self.min_year)
        if raw.empty:
            raise RuntimeError("Nessun match Sackmann caricato")
        clean = clean_matches(raw)
        clean = add_serve_stats(clean)
        odds = load_odds_history()
        if not odds.empty:
            clean = self._merge_odds(clean, odds)
        if save:
            PROCESSED.mkdir(parents=True, exist_ok=True)
            clean.to_csv(MATCHES_PATH, index=False)
        return clean

    def _merge_odds(self, matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
        """Join fuzzy su data + nomi giocatori."""
        o = odds.copy()
        date_col = next((c for c in ("Date", "date", "DATE") if c in o.columns), None)
        if not date_col:
            return matches
        o["match_date"] = _parse_odds_dates(o[date_col])
        w_col = next((c for c in ("Winner", "winner", "Winner") if c in o.columns), None)
        l_col = next((c for c in ("Loser", "loser", "Loser") if c in o.columns), None)
        if not w_col or not l_col:
            return matches

        ps_col = next((c for c in ("PSW", "B365W", "AvgW") if c in o.columns), None)
        pl_col = next((c for c in ("PSL", "B365L", "AvgL") if c in o.columns), None)
        if not ps_col:
            return matches

        o["odds_winner"] = pd.to_numeric(o[ps_col], errors="coerce")
        o["odds_loser"] = pd.to_numeric(o[pl_col], errors="coerce") if pl_col else None
        o["odds_key"] = vector_odds_match_keys(o["match_date"], o[w_col], o[l_col])
        o = o.dropna(subset=["odds_key"])
        if o.empty:
            return matches

        m = matches.copy()
        m["tourney_date"] = pd.to_datetime(m["tourney_date"], errors="coerce")
        m["odds_key"] = vector_odds_match_keys(
            m["tourney_date"],
            m["winner_name"],
            m["loser_name"],
        )
        merged = m.merge(
            o[["odds_key", "odds_winner", "odds_loser"]].drop_duplicates("odds_key"),
            on="odds_key",
            how="left",
        )
        return merged.drop(columns=["odds_key"], errors="ignore")


def load_matches() -> pd.DataFrame:
    if MATCHES_PATH.exists():
        return pd.read_csv(MATCHES_PATH, low_memory=False, parse_dates=["tourney_date"])
    return DatasetLoader().build()
