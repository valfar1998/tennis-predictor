"""Bridge verso infotennis: legge dati locali scrapati da atptour.com/Infosys."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "infotennis"


def _resolve_source() -> Path | None:
    env = os.environ.get("INFOTENNIS_PATH", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p
    local = Path(r"C:\Users\valba\Downloads\infotennis-main")
    return local if local.exists() else None


def sync_infotennis_data(*, copy: bool = False) -> dict:
    """Indicizza dati infotennis locali (keystats JSON, players CSV)."""
    src = _resolve_source()
    if not src:
        return {"ok": False, "error": "INFOTENNIS_PATH non configurato"}

    players_file = src / "data" / "players_ATP.csv"
    n_players = 0
    if players_file.exists():
        df = pd.read_csv(players_file)
        n_players = len(df)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "players_atp.csv", index=False)

    raw_dir = src / "data" / "key-stats" / "raw"
    n_json = sum(1 for _ in raw_dir.glob("**/*.json")) if raw_dir.exists() else 0

    return {
        "ok": True,
        "source": str(src),
        "n_players": n_players,
        "n_keystats_json": n_json,
        "note": "Esegui infotennis update_routines per popolare keystats (ATP 2021+)",
    }


def load_player_id_map() -> dict[str, str]:
    """Mappa nome giocatore -> ATP ID (es. DH58)."""
    path = OUT_DIR / "players_atp.csv"
    src = _resolve_source()
    if not path.exists() and src:
        pf = src / "data" / "players_ATP.csv"
        if pf.exists():
            path = pf
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    name_col = next((c for c in df.columns if "name" in c.lower()), df.columns[0])
    id_col = next((c for c in df.columns if "id" in c.lower()), df.columns[1])
    return {str(r[name_col]).strip().lower(): str(r[id_col]) for _, r in df.iterrows()}


def load_keystats_summary() -> pd.DataFrame:
    """Aggrega keystats JSON locali se presenti."""
    src = _resolve_source()
    if not src:
        return pd.DataFrame()
    raw_dir = src / "data" / "key-stats" / "raw"
    if not raw_dir.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for jf in raw_dir.glob("**/*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                rows.append(_parse_keystats(data, jf.stem))
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parse_keystats(data: dict, match_id: str) -> dict:
    """Estrae metriche chiave da JSON Infosys (struttura variabile)."""
    row = {"match_id": match_id}
    for key in ("serveRating", "aces", "doubleFaults", "firstServePct", "firstServeWonPct"):
        if key in data:
            row[key] = data[key]
    return row
