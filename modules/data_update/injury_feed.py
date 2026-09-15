"""Feed ritiri / infortuni / walkover da RSS + Reddit (gratis, no API enterprise).

Usa feed pubblici (ATP/WTA/ESPN/Tennis Abstract/BBC) e r/tennis JSON.
Matcha i titoli con i giocatori del palinsesto (`upcoming_predictions`) e
produce flag `news_alert` / boost `p_retire` / hard-block su withdrawal.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "injury_news.json"
CACHE_MAX_AGE_H = 2.0
UA = (
    "Mozilla/5.0 (compatible; TennisPredictor/1.0; +https://github.com/local) "
    "AppleWebKit/537.36"
)

# Feed RSS pubblici (best-effort: alcuni possono cambiare URL)
RSS_FEEDS: tuple[tuple[str, str], ...] = (
    ("espn_tennis", "https://www.espn.com/espn/rss/tennis/news"),
    ("bbc_tennis", "https://feeds.bbci.co.uk/sport/tennis/rss.xml"),
    ("tennis_abstract", "https://tennisabstract.com/blog/feed/"),
    ("atp_news", "https://www.atptour.com/en/media/rss-news"),
    ("wta_news", "https://www.wtatennis.com/rss-news-all-news.xml"),
    # Google News query (gratis): ritiri / injury focused
    (
        "gnews_injury_en",
        "https://news.google.com/rss/search?q=tennis+(withdrawal+OR+withdraw+OR+injury+OR+walkover+OR+retire)+when:7d&hl=en-US&gl=US&ceid=US:en",
    ),
    (
        "gnews_injury_it",
        "https://news.google.com/rss/search?q=tennis+(ritiro+OR+infortunio+OR+walkover+OR+forfait)+when:7d&hl=it&gl=IT&ceid=IT:it",
    ),
)

REDDIT_JSON = "https://www.reddit.com/r/tennis/new.json?limit=40"

# Keyword → severity (0–1) e categoria
_KEYWORD_RULES: tuple[tuple[str, str, float], ...] = (
    (r"\bwithdraw(?:al|s|n)?\b", "withdrawal", 0.95),
    (r"\bpull(?:s|ed)?\s+out\b", "withdrawal", 0.92),
    (r"\bscratch(?:ed)?\b", "withdrawal", 0.90),
    (r"\bforfait\b|\briti(?:ro|rarsi|rato)\b", "withdrawal", 0.92),
    (r"\bwalkover\b|\bw/?o\b", "walkover", 0.95),
    (r"\bretire(?:s|d|ment)?\b", "retirement", 0.85),
    (r"\binjur(?:y|ed|ies)\b|\binfortunio\b|\binfortunat[oa]\b", "injury", 0.75),
    (r"\bsidelined\b|\bout\s+with\b|\bmiss(?:es|ing)\s+(?:the\s+)?(?:tournament|match|final)", "injury", 0.72),
    (r"\bankle\b|\bknee\b|\bshoulder\b|\bhip\b|\belbow\b|\bwrist\b|\bhamstring\b|\babdominal\b", "injury", 0.55),
    (r"\bill(?:ness)?\b|\bsick\b|\bvirus\b|\bstomach\b|\bfood\s+poison|\bgastro\b", "illness", 0.70),
    (r"\bmedical\s+timeout\b|\bmto\b", "medical", 0.55),
    (r"\bskip(?:s|ping)?\s+(?:practice|training)\b|\bsalted\s+practice\b", "practice_skip", 0.60),
    (r"\bdoubt(?:ful)?\b|\bquestionable\b|\bfitness\s+doubt", "doubt", 0.45),
    (r"\bfitness\b|\bphysio\b|\btreatment\b|\brehab\b", "fitness", 0.40),
    (r"\bunavailable\b|\bnot\s+playing\b|\bout\s+of\s+(?:the\s+)?(?:draw|event)", "withdrawal", 0.80),
)

_HARD_BLOCK_CATS = frozenset({"withdrawal", "walkover"})
_BOOST_CATS = frozenset({"injury", "illness", "retirement", "medical", "practice_skip", "doubt", "fitness"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _http_get(url: str, *, timeout: float = 18.0) -> bytes | None:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/rss+xml, application/json, */*"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return None


def _strip_html(text: str) -> str:
    t = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", t).strip()


def _parse_rss_items(raw: bytes, *, source: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return items
    # RSS 2.0 o Atom
    nodes = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for node in nodes[:40]:
        title_el = node.find("title")
        if title_el is None:
            title_el = node.find("{http://www.w3.org/2005/Atom}title")
        title = (title_el.text or "").strip() if title_el is not None else ""

        link = ""
        link_el = node.find("link")
        if link_el is not None:
            link = (link_el.text or link_el.get("href") or "").strip()
        if not link:
            atom_link = node.find("{http://www.w3.org/2005/Atom}link")
            if atom_link is not None:
                link = (atom_link.get("href") or atom_link.text or "").strip()

        desc_el = (
            node.find("description")
            or node.find("{http://www.w3.org/2005/Atom}summary")
            or node.find("{http://www.w3.org/2005/Atom}content")
        )
        summary = _strip_html((desc_el.text or "") if desc_el is not None else "")

        pub_el = (
            node.find("pubDate")
            or node.find("{http://www.w3.org/2005/Atom}updated")
            or node.find("{http://www.w3.org/2005/Atom}published")
        )
        published = (pub_el.text or "").strip() if pub_el is not None else ""
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": link,
                "summary": summary[:500],
                "published": published,
                "source": source,
                "channel": "rss",
            }
        )
    return items


def _parse_reddit_items(raw: bytes) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return items
    for child in (data.get("data") or {}).get("children") or []:
        d = child.get("data") or {}
        title = str(d.get("title") or "").strip()
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": f"https://www.reddit.com{d.get('permalink') or ''}",
                "summary": _strip_html(str(d.get("selftext") or ""))[:400],
                "published": datetime.fromtimestamp(
                    float(d.get("created_utc") or 0), tz=timezone.utc
                ).isoformat()
                if d.get("created_utc")
                else "",
                "source": "reddit_r_tennis",
                "channel": "reddit",
            }
        )
    return items


def _classify(text: str) -> tuple[str | None, float]:
    t = text.lower()
    best_cat: str | None = None
    best_sev = 0.0
    for pattern, cat, sev in _KEYWORD_RULES:
        if re.search(pattern, t, flags=re.I):
            if sev > best_sev:
                best_cat, best_sev = cat, sev
    return best_cat, best_sev


def fetch_injury_news(*, force: bool = False, max_age_hours: float = CACHE_MAX_AGE_H) -> dict[str, Any]:
    """Scarica / riusa cache feed news tennis rilevanti per ritiri/infortuni."""
    if not force and CACHE.exists():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            ts = str(cached.get("fetched_at") or "")
            if ts:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                age_h = (_now() - fetched).total_seconds() / 3600
                if age_h < max_age_hours and cached.get("items") is not None:
                    return {**cached, "from_cache": True}
        except Exception:
            pass

    items: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, url in RSS_FEEDS:
        raw = _http_get(url)
        if not raw:
            errors.append(f"{name}: fetch failed")
            continue
        parsed = _parse_rss_items(raw, source=name)
        if not parsed:
            errors.append(f"{name}: empty/parse")
        items.extend(parsed)

    reddit_raw = _http_get(REDDIT_JSON)
    if reddit_raw:
        items.extend(_parse_reddit_items(reddit_raw))
    else:
        errors.append("reddit: fetch failed")

    classified: list[dict[str, Any]] = []
    for it in items:
        blob = f"{it.get('title') or ''} {it.get('summary') or ''}"
        cat, sev = _classify(blob)
        if not cat:
            continue
        classified.append({**it, "category": cat, "severity": sev})

    # Dedup per titolo normalizzato
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for it in sorted(classified, key=lambda x: float(x.get("severity") or 0), reverse=True):
        key = re.sub(r"\W+", " ", str(it.get("title") or "").lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(it)

    payload = {
        "ok": True,
        "fetched_at": _now().isoformat(),
        "n_items": len(unique),
        "n_raw": len(items),
        "errors": errors,
        "items": unique[:120],
        "from_cache": False,
    }
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_injury_news(*, force: bool = False) -> list[dict[str, Any]]:
    info = fetch_injury_news(force=force)
    return list(info.get("items") or [])


def _name_tokens(name: str) -> list[str]:
    from modules.data_update.entity_resolution import _last_name, _norm_name

    n = _norm_name(name)
    last = _last_name(name)
    tokens = [t for t in n.split() if len(t) >= 3]
    if last and last not in tokens and len(last) >= 3:
        tokens.append(last)
    return tokens


def _player_mentioned(name: str, text: str) -> bool:
    """Match robusto: cognome (+ iniziale se presente) nel testo news."""
    from modules.data_update.entity_resolution import _last_name, _norm_name

    if not name or not text:
        return False
    blob = _norm_name(text)
    last = _last_name(name)
    if not last or len(last) < 3:
        return False
    if last not in blob.split() and f" {last} " not in f" {blob} ":
        # cognome composto / truncate
        if last not in blob:
            return False
    # Evita falsi positivi su cognomi cortissimi già filtrati; se c'è solo cognome ok
    parts = [p for p in _norm_name(name).split() if len(p) >= 2]
    if len(parts) >= 2:
        # richiedi anche un altro token (iniziale o nome) se presente nel testo
        extras = [p for p in parts if p != last]
        if extras and not any(e in blob for e in extras):
            # ancora accettabile se cognome raro (>=6) e keyword injury nel titolo
            if len(last) < 6:
                return False
    return True


def match_news_to_players(
    player_names: list[str],
    *,
    news_items: list[dict[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Mappa nome giocatore → lista alert news rilevanti."""
    items = news_items if news_items is not None else load_injury_news(force=force)
    out: dict[str, list[dict[str, Any]]] = {n: [] for n in player_names}
    for name in player_names:
        hits: list[dict[str, Any]] = []
        for it in items:
            blob = f"{it.get('title') or ''} {it.get('summary') or ''}"
            if not _player_mentioned(name, blob):
                continue
            hits.append(
                {
                    "title": it.get("title"),
                    "link": it.get("link"),
                    "source": it.get("source"),
                    "category": it.get("category"),
                    "severity": it.get("severity"),
                    "published": it.get("published"),
                    "channel": it.get("channel"),
                }
            )
        hits.sort(key=lambda x: float(x.get("severity") or 0), reverse=True)
        out[name] = hits[:5]
    return out


def news_risk_for_player(hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggrega hit news → boost P(ritiro) e flag hard-block."""
    if not hits:
        return {
            "has_alert": False,
            "hard_block": False,
            "p_retire_boost": 0.0,
            "category": None,
            "headline": None,
            "n_hits": 0,
            "hits": [],
        }
    top = hits[0]
    cat = str(top.get("category") or "")
    sev = float(top.get("severity") or 0)
    hard = cat in _HARD_BLOCK_CATS and sev >= 0.85
    boost = 0.0
    if cat in _HARD_BLOCK_CATS:
        boost = min(0.25, 0.12 + sev * 0.12)
    elif cat in _BOOST_CATS:
        boost = min(0.18, sev * 0.18)
    return {
        "has_alert": True,
        "hard_block": hard,
        "p_retire_boost": round(boost, 4),
        "category": cat,
        "headline": top.get("title"),
        "link": top.get("link"),
        "source": top.get("source"),
        "severity": sev,
        "n_hits": len(hits),
        "hits": hits[:3],
    }


def enrich_predictions_with_news(
    predictions: list[dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Arricchisce predizioni con news_alert / news_a / news_b e modula p_retire."""
    if not predictions:
        return {"ok": True, "n_alerts": 0, "n_hard_blocks": 0}

    names: list[str] = []
    for p in predictions:
        for key in ("player_a", "player_b"):
            n = str(p.get(key) or "").strip()
            if n and n not in names:
                names.append(n)
        rn = p.get("resolved_names") or {}
        for key in ("a", "b"):
            n = str(rn.get(key) or "").strip()
            if n and n not in names:
                names.append(n)

    feed = fetch_injury_news(force=force)
    by_player = match_news_to_players(names, news_items=feed.get("items") or [])

    n_alerts = 0
    n_hard = 0
    for p in predictions:
        pa = str(p.get("player_a") or "")
        pb = str(p.get("player_b") or "")
        rn = p.get("resolved_names") or {}
        hits_a = list(by_player.get(pa) or [])
        hits_b = list(by_player.get(pb) or [])
        # Unisci anche nomi risolti
        for alt, bucket in ((rn.get("a"), "a"), (rn.get("b"), "b")):
            alt_s = str(alt or "").strip()
            if not alt_s:
                continue
            extra = by_player.get(alt_s) or []
            if bucket == "a":
                hits_a = hits_a + [h for h in extra if h not in hits_a]
            else:
                hits_b = hits_b + [h for h in extra if h not in hits_b]

        risk_a = news_risk_for_player(hits_a)
        risk_b = news_risk_for_player(hits_b)
        p["news_a"] = risk_a
        p["news_b"] = risk_b

        # Lato pick (best_play / recommended)
        from modules.advisor.advise import display_pick
        from modules.data_update.entity_resolution import _last_name

        rec = display_pick(p)
        pick = str(rec.get("player") or "")
        pick_side = "A"
        if pick and _last_name(pick) == _last_name(pb):
            pick_side = "B"
        pick_risk = risk_a if pick_side == "A" else risk_b
        opp_risk = risk_b if pick_side == "A" else risk_a

        alert = {
            "pick_side": pick_side,
            "pick": pick_risk,
            "opponent": opp_risk,
            "any_alert": bool(risk_a.get("has_alert") or risk_b.get("has_alert")),
            "hard_block_pick": bool(pick_risk.get("hard_block")),
            "headline": pick_risk.get("headline") or opp_risk.get("headline"),
            "category": pick_risk.get("category") or opp_risk.get("category"),
        }
        p["news_alert"] = alert
        if alert["any_alert"]:
            n_alerts += 1

        # Boost p_retire sul contesto retirement già presente
        boost_pick = float(pick_risk.get("p_retire_boost") or 0)
        boost_opp = float(opp_risk.get("p_retire_boost") or 0) * 0.35  # opposite side: meno critico
        boost = max(boost_pick, boost_opp)
        if boost > 0:
            base = float(p.get("p_retire") or (p.get("retirement_context") or {}).get("p_retire") or 0)
            new_p = min(0.45, base + boost)
            p["p_retire"] = round(new_p, 4)
            ctx = dict(p.get("retirement_context") or {})
            ctx["player_injury_risk"] = new_p
            ctx["p_retire"] = new_p
            ctx["news_boost"] = boost
            p["retirement_context"] = ctx

        # Hard-block: withdrawal sul pick → no_bet (anche shadow/bet)
        if alert["hard_block_pick"]:
            n_hard += 1
            reasons = [
                f"news: {pick_risk.get('category')} — {pick_risk.get('headline') or 'ritiro/infortunio segnalato'}"
            ]
            p["action"] = "no_bet"
            p["recommended"] = None
            p["shadow"] = False
            bp = dict(p.get("best_play") or rec or {})
            bp["action"] = "no_bet"
            bp["no_bet_reasons"] = list(bp.get("no_bet_reasons") or []) + reasons
            bp["kelly"] = 0.0
            p["best_play"] = bp
            p["news_hard_block"] = True
            # Cap playability
            if p.get("playability") is not None:
                p["playability"] = min(float(p["playability"]), 35.0)
                p["playability_band"] = "no_bet"
                p["playability_label"] = "No bet"

        # Soft: injury sul pick → review se era bet
        elif (
            pick_risk.get("has_alert")
            and str(pick_risk.get("category") or "") in _BOOST_CATS
            and p.get("action") == "bet"
            and float(pick_risk.get("severity") or 0) >= 0.70
        ):
            p["action"] = "review"
            rec2 = dict(p.get("recommended") or p.get("best_play") or {})
            rec2["action"] = "review"
            rec2["review_reasons"] = list(rec2.get("review_reasons") or []) + [
                f"news review: {pick_risk.get('category')} — {pick_risk.get('headline')}"
            ]
            p["recommended"] = rec2
            p["best_play"] = rec2

    return {
        "ok": True,
        "n_alerts": n_alerts,
        "n_hard_blocks": n_hard,
        "feed_items": int(feed.get("n_items") or 0),
        "from_cache": feed.get("from_cache"),
        "fetched_at": feed.get("fetched_at"),
        "errors": feed.get("errors") or [],
    }


def format_news_banner(enrich_info: dict[str, Any] | None = None) -> str:
    info = enrich_info or {}
    return (
        f"News/injury feed: {info.get('n_alerts', 0)} alert su palinsesto, "
        f"{info.get('n_hard_blocks', 0)} hard-block withdrawal "
        f"(feed {info.get('feed_items', 0)} item)"
    )
