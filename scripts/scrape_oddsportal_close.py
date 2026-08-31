#!/usr/bin/env python3
"""CLI: popola data/raw/oddsportal_close.json da OddsPortal (Playwright)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape quote chiusura OddsPortal → oddsportal_close.json")
    parser.add_argument("--max-matches", type=int, default=80, help="Max match da processare")
    parser.add_argument("--no-h2h", action="store_true", help="Solo pagina results (più veloce, meno preciso)")
    parser.add_argument("--headed", action="store_true", help="Browser visibile (debug)")
    parser.add_argument("--url", action="append", dest="urls", help="URL torneo /results/ (ripetibile)")
    parser.add_argument("--year", type=int, action="append", dest="years", help="Anni torneo (default: corrente e precedente)")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from modules.data_update.oddsportal_scraper import default_tournament_urls, scrape_oddsportal_close

    urls = args.urls
    if not urls and args.years:
        urls = default_tournament_urls(years=args.years)

    result = scrape_oddsportal_close(
        tournament_urls=urls,
        max_matches=args.max_matches,
        enrich_h2h=not args.no_h2h,
        headless=not args.headed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
