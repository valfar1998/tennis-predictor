"""Entity resolution: normalizzazione nomi giocatori e fuzzy matching."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[2]
ALIASES_PATH = ROOT / "data" / "processed" / "player_aliases.json"


def _norm_name(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canonical_key(name: str) -> str:
    """Chiave canonica: cognome + iniziale nome."""
    parts = _norm_name(name).split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-1]} {parts[0][0]}"


def load_aliases() -> dict[str, str]:
    if not ALIASES_PATH.exists():
        return {}
    try:
        return json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_aliases(aliases: dict[str, str]) -> None:
    ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIASES_PATH.write_text(json.dumps(aliases, indent=2, ensure_ascii=False), encoding="utf-8")


def build_player_index(players_df: pd.DataFrame) -> dict[int, str]:
    """Mappa player_id → nome completo."""
    idx: dict[int, str] = {}
    for _, row in players_df.iterrows():
        pid = row.get("player_id")
        if pd.isna(pid):
            continue
        first = str(row.get("name_first") or "").strip()
        last = str(row.get("name_last") or "").strip()
        name = f"{first} {last}".strip()
        if name:
            idx[int(pid)] = name
    return idx


def _resolve_initial_lastname(name: str, candidates: list[str]) -> str | None:
    """Risolve formati Betfair tipo 'H Dart' o 'Pe Stearns'."""
    parts = str(name or "").replace(".", " ").split()
    if len(parts) < 2:
        return None
    prefix, last = parts[0].lower(), parts[-1].lower()
    if len(last) < 3 or len(prefix) > 3:
        return None

    def _matches(cand: str) -> bool:
        cp = cand.replace(".", " ").split()
        if not cp:
            return False
        cand_last = cp[-1].lower() if len(cp[-1]) > 2 else cp[0].lower()
        if cand_last != last:
            return False
        first = cp[0].lower()
        if prefix == first:
            return True
        if first.startswith(prefix):
            return True
        if prefix.startswith(first[: max(1, len(prefix))]):
            return True
        return first[:1] == prefix[:1]

    hits = [c for c in candidates if _matches(c)]
    if len(hits) == 1:
        return hits[0]
    return None


def _resolve_unique_lastname(name: str, candidates: list[str]) -> str | None:
    parts = str(name or "").replace(".", " ").split()
    if len(parts) != 1:
        return None
    last = parts[0].lower()
    hits = [c for c in candidates if _last_name(c) == last]
    if len(hits) == 1:
        return hits[0]
    return None


def resolve_name(
    name: str,
    *,
    candidates: list[str] | None = None,
    aliases: dict | None = None,
    opponent_name: str | None = None,
    tourney_date: str | None = None,
    tour: str | None = None,
) -> str:
    """Risolve un nome: graph → registry SQLite → alias JSON → fuzzy."""
    try:
        from modules.data_update.player_graph import graph_resolve_player

        graph_hit = graph_resolve_player(
            name,
            tour=tour,
            opponent_name=opponent_name,
            tourney_date=tourney_date,
        )
        if graph_hit:
            return graph_hit
    except Exception:
        pass

    try:
        from modules.data_update.player_registry import resolve_canonical

        canon = resolve_canonical(name)
        if canon:
            return canon
    except Exception:
        pass

    aliases = aliases or load_aliases()
    norm = _norm_name(name)
    if norm in aliases:
        return aliases[norm]

    key = _canonical_key(name)
    if key in aliases:
        return aliases[key]

    if candidates:
        by_init = _resolve_initial_lastname(name, candidates)
        if by_init:
            return by_init
        by_last = _resolve_unique_lastname(name, candidates)
        if by_last:
            return by_last
        match = process.extractOne(name, candidates, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 88:
            return match[0]
        qparts = str(name).replace(".", " ").split()
        if len(qparts) >= 2 and len(qparts[0]) <= 3:
            pref = qparts[0].lower()
            same_last = [c for c in candidates if _last_name(c) == _last_name(name)]
            pref_hits = [c for c in same_last if c.split()[0].lower().startswith(pref)]
            if len(pref_hits) == 1:
                return pref_hits[0]
        match = process.extractOne(name, candidates, scorer=fuzz.WRatio)
        if match and match[1] >= 82 and _last_name(match[0]) == _last_name(name):
            return match[0]

    return str(name).strip()


def _last_name(name: str) -> str:
    """Estrae cognome da 'Novak Djokovic' o 'Djokovic N.'."""
    s = str(name or "").strip()
    if not s:
        return ""
    parts = s.replace(".", "").split()
    if len(parts) == 1:
        return parts[0].lower()
    # Formato tennis-data: "Djokovic N" -> cognome first
    if len(parts[-1]) <= 2:
        return parts[0].lower()
    return parts[-1].lower()


def _last_name_series(names: pd.Series) -> pd.Series:
    """Versione vettorizzata di _last_name (cache su valori unici)."""
    s = names.fillna("").astype(str).str.strip()
    uniques = s.unique()
    mapping = {u: _last_name(u) for u in uniques if u}
    return s.map(mapping).fillna("")


def vector_odds_match_keys(
    dates: pd.Series,
    winners: pd.Series,
    losers: pd.Series,
) -> pd.Series:
    """Chiavi join odds/match; NaN dove data o nomi invalidi."""
    d = pd.to_datetime(dates, errors="coerce")
    w = winners.fillna("").astype(str).str.strip()
    l = losers.fillna("").astype(str).str.strip()
    bad = {"nan", "none", ""}
    valid = d.notna() & ~w.str.lower().isin(bad) & ~l.str.lower().isin(bad)
    keys = pd.Series(pd.NA, index=dates.index, dtype=object)
    if not valid.any():
        return keys
    dstr = d.dt.strftime("%Y-%m-%d")
    keys.loc[valid] = (
        dstr[valid] + "|" + _last_name_series(w[valid]) + "|" + _last_name_series(l[valid])
    )
    return keys


def odds_match_key(date_str: str, winner: str, loser: str) -> str:
    return f"{date_str}|{_last_name(winner)}|{_last_name(loser)}"


def match_players_to_odds(
    odds_names: list[str],
    sackmann_names: list[str],
    *,
    threshold: int = 85,
) -> dict[str, str]:
    """Mappa nomi da tennis-data.co.uk verso nomi Sackmann."""
    aliases = load_aliases()
    mapping: dict[str, str] = {}
    for odds_name in odds_names:
        resolved = resolve_name(odds_name, candidates=sackmann_names, aliases=aliases)
        if resolved != odds_name:
            mapping[odds_name] = resolved
            continue
        match = process.extractOne(odds_name, sackmann_names, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            mapping[odds_name] = match[0]
    return mapping


def player_side_match(a: str, b: str) -> bool:
    """True se a e b indicano lo stesso giocatore (cognome / fuzzy leggero)."""
    a_n, b_n = _norm_name(a), _norm_name(b)
    if not a_n or not b_n:
        return False
    if a_n == b_n:
        return True
    if _last_name(a) and _last_name(a) == _last_name(b):
        return True
    if len(a_n) >= 4 and len(b_n) >= 4 and (a_n in b_n or b_n in a_n):
        return True
    return False


def align_odds_to_players(
    player_a: str,
    player_b: str,
    odd_a: float | None,
    odd_b: float | None,
    *,
    runner_a: str | None = None,
    runner_b: str | None = None,
) -> dict:
    """Garantisce odd_a → player_a e odd_b → player_b (swap se runner invertiti)."""
    oa = float(odd_a) if odd_a and float(odd_a) > 1.01 else None
    ob = float(odd_b) if odd_b and float(odd_b) > 1.01 else None
    out = {
        "odd_a": oa,
        "odd_b": ob,
        "swapped": False,
        "verified": False,
        "blocked": oa is None or ob is None,
    }
    if out["blocked"]:
        return out

    ra = str(runner_a or player_a).strip()
    rb = str(runner_b or player_b).strip()
    direct = player_side_match(ra, player_a) and player_side_match(rb, player_b)
    swap = player_side_match(ra, player_b) and player_side_match(rb, player_a)

    if swap and not direct:
        out["odd_a"] = ob
        out["odd_b"] = oa
        out["swapped"] = True
        out["verified"] = True
    elif direct:
        out["verified"] = True
    elif player_side_match(player_a, ra) or player_side_match(player_b, rb):
        out["verified"] = True
    else:
        out["blocked"] = True
        out["reason"] = "runner non allineati a player_a/player_b"

    return out


def detect_model_odds_inversion(
    p_win_a: float,
    odd_a: float | None,
    odd_b: float | None,
    *,
    prob_gap: float = 0.12,
    odds_ratio: float = 1.15,
) -> bool:
    """True se il favorito modello ha la quota più alta (probabile inversione)."""
    if not odd_a or not odd_b:
        return False
    oa, ob = float(odd_a), float(odd_b)
    p = float(p_win_a)
    if p >= 0.5 + prob_gap and oa > ob * odds_ratio:
        return True
    if p <= 0.5 - prob_gap and oa * odds_ratio < ob:
        return True
    return False


def dataset_match_count(player_id: int | None, matches: pd.DataFrame | None) -> int:
    """Match registrati TML/Sackmann per giocatore."""
    if player_id is None or matches is None or matches.empty:
        return 0
    pid = int(player_id)
    w = (matches["winner_id"] == pid).sum()
    l = (matches["loser_id"] == pid).sum()
    return int(w + l)


def shrink_rating_low_sample(
    rating: float,
    n_matches: int,
    *,
    prior: float | None = None,
) -> float:
    """Shrink Elo verso prior se pochi match in dataset storico."""
    from modules.constants import DATASET_LOW_SAMPLE, DATASET_PRIOR_WEIGHT, ELO_START

    if n_matches >= DATASET_LOW_SAMPLE:
        return rating
    p = prior if prior is not None else ELO_START
    k = DATASET_PRIOR_WEIGHT
    n = max(0, int(n_matches))
    return (n * rating + k * p) / (n + k)


def enrich_pressure_profile(profile: dict | None, player_id: int | None, matches: pd.DataFrame | None) -> dict:
    """Aggiunge n_dataset per shrinkage bayesiano su low sample."""
    out = dict(profile or {})
    n_ds = dataset_match_count(player_id, matches)
    out["n_dataset"] = n_ds
    n_mcp = int(out.get("n_matches") or out.get("n_charted") or 0)
    if n_mcp and n_ds:
        out["n_effective"] = min(n_mcp, n_ds)
    else:
        out["n_effective"] = n_mcp or n_ds
    return out
