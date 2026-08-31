"""Arricchimento giocatori via Wikidata SPARQL (gratuito, senza API key)."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "processed" / "wikidata_players.json"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
UA = "tennis-predictor/1.0 (educational; mailto:local@predictor)"


def _sparql(query: str, timeout: int = 30) -> list[dict]:
    url = SPARQL_ENDPOINT + "?" + urlencode({"query": query, "format": "json"})
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("results", {}).get("bindings", [])


def enrich_players_from_wikidata(
    players_df: pd.DataFrame,
    *,
    limit: int = 500,
    force: bool = False,
) -> dict:
    """Arricchisce giocatori ATP con altezza, mano, paese da Wikidata."""
    if not force and is_fresh(CACHE_PATH, max_age_hours=336):
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {"ok": True, "n_players": len(cached), "from_cache": True}

    df = players_df.dropna(subset=["wikidata_id"]).head(limit)
    if df.empty:
        return {"ok": False, "error": "nessun wikidata_id nei player"}

    enriched: dict[str, dict] = {}
    if CACHE_PATH.exists() and not force:
        try:
            enriched = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            enriched = {}

    ids = [f"wd:Q{int(row['wikidata_id'])}" for _, row in df.iterrows() if pd.notna(row.get("wikidata_id"))]
    if not ids:
        return {"ok": False, "error": "nessun ID valido"}

    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        values = " ".join(batch)
        query = f"""
        SELECT ?player ?playerLabel ?height ?hand ?countryLabel WHERE {{
          VALUES ?player {{ {values} }}
          OPTIONAL {{ ?player wdt:P2048 ?height. }}
          OPTIONAL {{ ?player wdt:P552 ?hand. }}
          OPTIONAL {{ ?player wdt:P27 ?country. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,it". }}
        }}
        """
        try:
            rows = _sparql(query)
            for row in rows:
                qid = row.get("player", {}).get("value", "").split("/")[-1]
                enriched[qid] = {
                    "name": row.get("playerLabel", {}).get("value"),
                    "height_cm": _parse_height(row.get("height", {}).get("value")),
                    "hand": row.get("hand", {}).get("value", "").split("/")[-1],
                    "country": row.get("countryLabel", {}).get("value"),
                }
        except Exception:
            continue

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "n_players": len(enriched), "from_cache": False}


def _parse_height(val: str | None) -> float | None:
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def lookup_player(qid: str | int | None) -> dict | None:
    if not qid:
        return None
    key = f"Q{int(str(qid).replace('Q', ''))}"
    if not CACHE_PATH.exists():
        return None
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return data.get(key)
