"""Policy di cache per download e scraping."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def is_fresh(path: Path, *, max_age_hours: float = 72.0) -> bool:
    if not path.exists():
        return False
    age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0
    return age_h < max_age_hours
