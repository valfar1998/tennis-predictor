"""Scraper Playwright OddsPortal — quote di chiusura Pinnacle (o fallback bookmaker)."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

from modules.data_update.oddsportal_close import CACHE, _load, save_close_rows
from modules.data_update.entity_resolution import odds_match_key

BASE = os.environ.get("ODDSPORTAL_BASE_URL", "https://www.oddsportal.com")

# Bookmaker preferiti per proxy CLV Pinnacle (ordine priorità)
PREFERRED_BOOKMAKERS = (
    "Pinnacle",
    "Pinnacle.com",
    "Betfair",
    "bet365",
    "bet365.it",
    "Marathon",
)

IT_MONTHS = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
    "jan": 1, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "dec": 12,
}

TOURNAMENT_PATHS = (
    "tennis/usa/atp-us-open-{year}/results/",
    "tennis/australia/atp-australian-open-{year}/results/",
    "tennis/france/atp-french-open-{year}/results/",
    "tennis/united-kingdom/atp-wimbledon-{year}/results/",
    "tennis/usa/wta-us-open-{year}/results/",
    "tennis/france/wta-open-di-francia-{year}/results/",
    "tennis/united-kingdom/wta-wimbledon-{year}/results/",
    "tennis/australia/wta-australian-open-{year}/results/",
    "tennis/atp-singles/{year}/results/",
)


def default_tournament_urls(*, years: list[int] | None = None) -> list[str]:
    y = years or [datetime.now().year, datetime.now().year - 1]
    urls: list[str] = []
    for year in y:
        for path in TOURNAMENT_PATHS:
            urls.append(f"{BASE}/{path.format(year=year)}")
    return urls


def _parse_it_date(text: str) -> str | None:
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})", text)
    if not m:
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", text)
        if m:
            d, mo, y = m.groups()
            return f"{y}-{mo}-{d}"
        return None
    day, mon, year = m.groups()
    mo = IT_MONTHS.get(mon.lower()[:3])
    if not mo:
        return None
    return f"{year}-{mo:02d}-{int(day):02d}"


def _accept_cookies(page) -> None:
    for sel in (
        "#onetrust-accept-btn-handler",
        "button:has-text('Accetta')",
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
    ):
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=1500):
                loc.click()
                page.wait_for_timeout(800)
                return
        except Exception:
            continue


def _parse_results_rows(page) -> list[dict[str, Any]]:
    """Estrae match finiti dalla pagina /results/."""
    return page.evaluate(
        r"""() => {
        const out = [];
        document.querySelectorAll('div.border-black-borders').forEach(row => {
          const text = row.innerText || '';
          if (!/Finita|Finished/i.test(text)) return;
          const link = row.querySelector('a[href*="h2h"]');
          if (!link) return;
          const odds = [...text.matchAll(/\b([1-9]\d*\.\d{2})\b/g)].map(m => parseFloat(m[1]))
            .filter(x => x >= 1.01 && x <= 100);
          if (odds.length < 2) return;

          const players = [];
          link.querySelectorAll('p, span, strong').forEach(el => {
            const t = (el.innerText || '').trim();
            if (!t || t.length < 4 || /Finita|Finished|FIN|^-$/.test(t)) return;
            if (!/[A-ZÀ-Ö][a-zà-ö'.-]/.test(t)) return;
            if (players.some(p => p.name === t)) return;
            const fw = getComputedStyle(el).fontWeight;
            const bold = fw === '700' || fw === 'bold' || (el.className || '').includes('font-bold');
            players.push({name: t, bold});
          });
          if (players.length < 2) return;
          const p1 = players[0].name;
          const p2 = players[1].name;
          let winner = p1, loser = p2, psw = odds[0], psl = odds[1];
          if (players[1].bold && !players[0].bold) {
            winner = p2; loser = p1; psw = odds[1]; psl = odds[0];
          } else if (players[0].bold && !players[1].bold) {
            winner = p1; loser = p2;
          }

          out.push({
            href: link.href,
            player_a: p1,
            player_b: p2,
            winner,
            loser,
            psw,
            psl,
            bookmaker: 'oddsportal_results',
          });
        });
        return out;
    }"""
    )


def _parse_bookmaker_odds(page) -> list[dict[str, Any]]:
    """Legge tabella bookmaker sulla pagina match/h2h."""
    return page.evaluate(
        r"""(preferred) => {
        const rows = [];
        const pushRow = (name, o1, o2) => {
          if (!name || !o1 || !o2) return;
          rows.push({bookmaker: name.trim(), odd_a: o1, odd_b: o2});
        };
        document.querySelectorAll('div.border-black-borders, tr').forEach(el => {
          const t = el.innerText || '';
          const nums = [...t.matchAll(/\b([1-9]\d*\.\d{2})\b/g)].map(m => parseFloat(m[1]))
            .filter(x => x >= 1.01 && x <= 100);
          if (nums.length < 2) return;
          const lines = t.split('\n').map(s => s.trim()).filter(Boolean);
          const name = lines.find(l => l.length > 2 && !/^[\\d.%]+$/.test(l) && !/bonus|richiedi/i.test(l));
          if (name) pushRow(name, nums[0], nums[1]);
        });
        return rows;
    }""",
        PREFERRED_BOOKMAKERS,
    )


def _pick_bookmaker(rows: list[dict], *, player_a: str, player_b: str, winner: str) -> dict | None:
    if not rows:
        return None
    winner_is_a = winner.strip().lower() in player_a.strip().lower() or player_a.strip().lower() in winner.strip().lower()

    def map_odds(row: dict) -> tuple[float, float]:
        oa, ob = float(row["odd_a"]), float(row["odd_b"])
        if winner_is_a:
            return oa, ob
        return ob, oa

    for pref in PREFERRED_BOOKMAKERS:
        for row in rows:
            bk = str(row.get("bookmaker") or "")
            if pref.lower() in bk.lower():
                psw, psl = map_odds(row)
                return {"psw": psw, "psl": psl, "bookmaker": bk}
    # mediana su tutti i bookmaker
    psws, psls = [], []
    for row in rows:
        psw, psl = map_odds(row)
        psws.append(psw)
        psls.append(psl)
    psws.sort()
    psls.sort()
    mid = len(psws) // 2
    return {"psw": psws[mid], "psl": psls[mid], "bookmaker": "oddsportal_median"}


def _enrich_from_h2h(page, match: dict[str, Any], *, timeout_ms: int = 25000) -> dict[str, Any]:
    href = match.get("href")
    if not href:
        return match
    try:
        page.goto(href, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(4000)
        body = page.inner_text("body")
        if len(body) < 500:
            page.wait_for_timeout(4000)
            body = page.inner_text("body")

        date = _parse_it_date(body)
        if date:
            match["date"] = date

        bk_rows = _parse_bookmaker_odds(page)
        picked = _pick_bookmaker(
            bk_rows,
            player_a=str(match.get("player_a") or ""),
            player_b=str(match.get("player_b") or ""),
            winner=str(match.get("winner") or ""),
        )
        if picked:
            match["psw"] = picked["psw"]
            match["psl"] = picked["psl"]
            match["bookmaker"] = picked["bookmaker"]
    except Exception as exc:
        match["scrape_error"] = str(exc)[:120]
    return match


def merge_rows(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Unisce per chiave data+vincitore+perdente (ultimo vince)."""
    merged: dict[str, dict] = {}
    for row in existing + new_rows:
        day = str(row.get("date") or "")[:10]
        w, l = str(row.get("winner") or ""), str(row.get("loser") or "")
        if not w or not l:
            continue
        key = odds_match_key(day, w, l) if day else f"{w}|{l}"
        merged[key] = {
            "date": day or row.get("date"),
            "winner": w,
            "loser": l,
            "psw": row.get("psw"),
            "psl": row.get("psl"),
            "bookmaker": row.get("bookmaker"),
        }
    return list(merged.values())


def scrape_oddsportal_close(
    *,
    tournament_urls: list[str] | None = None,
    max_matches: int = 80,
    enrich_h2h: bool = True,
    headless: bool = True,
    delay_s: float = 0.8,
) -> dict[str, Any]:
    """Scrape quote di chiusura da OddsPortal → oddsportal_close.json."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return {"ok": False, "error": "playwright non installato (pip install playwright && playwright install chromium)"}

    urls = tournament_urls or default_tournament_urls()
    scraped: list[dict] = []
    errors: list[str] = []

    with sync_playwright() as p:
        launch_kwargs: dict[str, Any] = {"headless": headless}
        proxy_url = os.environ.get("ODDSPORTAL_PROXY", "").strip()
        if proxy_url:
            launch_kwargs["proxy"] = {"server": proxy_url}

        browser = p.chromium.launch(**launch_kwargs)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            locale=os.environ.get("ODDSPORTAL_LOCALE", "en-GB"),
            extra_http_headers={"Accept-Language": "en-GB,en;q=0.9"},
        )
        page = ctx.new_page()
        cookies_done = False

        for url in urls:
            if len(scraped) >= max_matches:
                break
            try:
                page.goto(url, wait_until="networkidle", timeout=90000)
                if not cookies_done:
                    _accept_cookies(page)
                    cookies_done = True
                page.wait_for_timeout(3000)
                rows = _parse_results_rows(page)
                for row in rows:
                    if len(scraped) >= max_matches:
                        break
                    if enrich_h2h:
                        row = _enrich_from_h2h(page, row)
                        time.sleep(delay_s)
                    if not row.get("date"):
                        # fallback: estrai anno dal URL torneo
                        ym = re.search(r"-(\d{4})/", url)
                        row["date"] = f"{ym.group(1)}-01-01" if ym else ""
                    scraped.append({
                        "date": row.get("date"),
                        "winner": row.get("winner"),
                        "loser": row.get("loser"),
                        "psw": row.get("psw"),
                        "psl": row.get("psl"),
                        "bookmaker": row.get("bookmaker"),
                    })
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        browser.close()

    existing = _load().get("rows") or []
    merged = merge_rows(existing, scraped)
    save = save_close_rows(merged)

    return {
        "ok": True,
        "scraped": len(scraped),
        "total_rows": len(merged),
        "path": str(CACHE),
        "errors": errors[:10],
        **save,
    }
