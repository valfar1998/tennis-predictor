"""Config calibrazione e soglie."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAL_PATH = ROOT / "data" / "models" / "calibration.json"

DEFAULTS = {
    "min_ev_play": 0.025,
    "kelly_fraction": 0.20,
    "kelly_cap": 0.018,
    "min_prob_play": 0.38,
    "devig_method": "shin",
    "reliability_ml": [],
    "backtest_summary": {},
    "by_surface": [],
    "by_level": [],
}


def load_calibration(*, force: bool = False) -> dict:
    if not CAL_PATH.exists():
        return dict(DEFAULTS)
    data = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    return {**DEFAULTS, **data}


def save_calibration(data: dict) -> Path:
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAL_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return CAL_PATH
