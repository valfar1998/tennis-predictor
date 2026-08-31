"""Live scores da livescore.tennis-data.co.uk (widget LiveXscores)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "tennis_livescore.json"
LIVESCORE_URL = "http://livescore.tennis-data.co.uk/"
LIVEX_URL = "https://www.livexscores.com/free2.php?p=0&sport=tennis&style=x"

UA = "Mozilla/5.0 (compatible; tennis-predictor/1.0)"


def _fetch_html(url: str) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": UA, "Referer": LIVESCORE_URL})
        with urlopen(req, timeout=25) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


def _parse_livescore_html(html: str) -> list[dict]:
    """Estrae match da HTML LiveXscores (best-effort)."""
    rows: list[dict] = []
    # pattern tipico: giocatore - giocatore con score
    for block in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.I | re.S):
        text = re.sub(r"<[^>]+>", " ", block)
        text = re.sub(r"\s+", " ", text).strip()
        if " - " not in text and " v " not in text.lower():
            continue
        if len(text) < 8:
            continue
        rows.append({"raw": text[:200]})
    return rows[:100]


def fetch_tennis_livescore(*, force: bool = False) -> dict:
    """Aggiorna cache livescore (può fallire se LiveXscores blocca bot)."""
    if not force and is_fresh(CACHE, max_age_hours=0.25) and CACHE.is_file():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass

    html = _fetch_html(LIVEX_URL)
    info: dict = {
        "ok": False,
        "source": LIVESCORE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "matches": [],
    }
    if html and "cloudflare" not in html.lower()[:500]:
        parsed = _parse_livescore_html(html)
        if parsed:
            info["ok"] = True
            info["matches"] = parsed
            info["n_matches"] = len(parsed)
        else:
            info["note"] = "HTML ricevuto ma nessun match parsato"
    else:
        info["error"] = "LiveXscores non disponibile (Cloudflare o timeout)"
        info["fallback"] = "Usa Betfair + tennis-data quote storiche"

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info
