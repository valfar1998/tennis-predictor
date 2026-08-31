"""Settle pick pendenti + online learn (job giornaliero o manuale)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from modules.data_update.history import settle_pending

    out = settle_pending(learn=True)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
