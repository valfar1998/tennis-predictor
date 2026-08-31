"""Feed calendario ATP/WTA — palinsesto e risultati recenti.

Nota: atptour.com e wtatennis.com usano Cloudflare; per scraping completo
usare infotennis (Selenium) o seeder (TennisExplorer). Qui: cache locale + fallback.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE_ATP = ROOT / "data" / "raw" / "atp_calendar.json"
CACHE_WTA = ROOT / "data" / "raw" / "wta_calendar.json"
UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"

# Endpoint pubblici alternativi (meno protetti)
ATP_RANKINGS_URL = "https://www.atptour.com/-/api/rankings/rankingsData"


def fetch_atp_rankings(*, force: bool = False) -> dict:
    """Prova API rankings ATP (spesso bloccata da Cloudflare)."""
    if not force and is_fresh(CACHE_ATP, max_age_hours=24):
        data = json.loads(CACHE_ATP.read_text(encoding="utf-8"))
        return {"ok": True, "n_players": len(data.get("rankings", [])), "from_cache": True}

    try:
        resp = requests.get(ATP_RANKINGS_URL, headers={"User-Agent": UA, "Accept": "application/json"}, timeout=20)
        if resp.status_code != 200:
            return {"ok": False, "error": f"ATP API status {resp.status_code} — usa infotennis/seeder"}
        data = resp.json()
        payload = {"rankings": data, "fetched_at": datetime.utcnow().isoformat()}
        CACHE_ATP.parent.mkdir(parents=True, exist_ok=True)
        CACHE_ATP.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return {"ok": True, "n_players": len(data) if isinstance(data, list) else 0, "from_cache": False}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def load_upcoming_from_sackmann(*, days_ahead: int = 14) -> pd.DataFrame:
    """Fallback: ultimi tornei in corso da dati Sackmann."""
    from modules.data_update.sackmann import load_tour_matches

    matches = load_tour_matches(min_year=datetime.now().year - 1)
    if matches.empty:
        return pd.DataFrame()
    cutoff = datetime.now() - timedelta(days=7)
    recent = matches[pd.to_datetime(matches["tourney_date"]) >= cutoff]
    return recent.drop_duplicates(subset=["tourney_name", "winner_name", "loser_name"])


def sync_tour_feeds(*, force: bool = False) -> dict:
    """Sincronizza feed tour disponibili."""
    atp = fetch_atp_rankings(force=force)
    wta = {"ok": False, "note": "WTA richiede infotennis/seeder — wtatennis.com protetto da Cloudflare"}
    return {"atp": atp, "wta": wta}
