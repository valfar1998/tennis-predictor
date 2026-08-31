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


def resolve_name(name: str, *, candidates: list[str] | None = None, aliases: dict | None = None) -> str:
    """Risolve un nome verso la forma canonica usando alias e fuzzy match."""
    aliases = aliases or load_aliases()
    norm = _norm_name(name)
    if norm in aliases:
        return aliases[norm]

    key = _canonical_key(name)
    if key in aliases:
        return aliases[key]

    if candidates:
        match = process.extractOne(name, candidates, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 88:
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
