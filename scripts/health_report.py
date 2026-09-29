#!/usr/bin/env python3
"""CLI: report salute modello (volume bet, Telegram, BCR/close).

  python scripts/health_report.py
  python scripts/health_report.py --days 21 --refresh-metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Tennis Predictor — health report")
    parser.add_argument("--days", type=int, default=14, help="Finestra volume giornaliero")
    parser.add_argument(
        "--refresh-metrics",
        action="store_true",
        help="Ricalcola live_metrics (BCR) prima del report",
    )
    args = parser.parse_args()

    from modules.advisor.health_report import build_health_report, format_health_banner

    report = build_health_report(days=args.days, refresh_metrics=args.refresh_metrics)
    print(format_health_banner(report))
    print(json.dumps(report.get("summary") or {}, indent=2, ensure_ascii=False))
    print(f"Report: data/processed/health_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
