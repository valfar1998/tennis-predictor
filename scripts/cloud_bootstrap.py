"""Allena modello su GitHub Actions (TML + Sackmann + tennis-data.co.uk)."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# openpyxl: xlsx tennis-data.co.uk contengono estensioni ignote (non influiscono sui dati)
warnings.filterwarnings(
    "ignore",
    message="Unknown extension is not supported and will be removed",
    category=UserWarning,
)

MIN_YEAR = 2010


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from scripts.run_retrain_pipeline import run_full_ml_pipeline

    result = run_full_ml_pipeline(min_year=MIN_YEAR, force_features=True)
    print(json.dumps(result, indent=2, default=str))

    model = ROOT / "data" / "models" / "best_model.joblib"
    stacker = ROOT / "data" / "models" / "meta_learner.joblib"
    if not model.is_file():
        raise SystemExit("bootstrap: manca data/models/best_model.joblib")
    if not stacker.is_file() and not (ROOT / "data" / "models" / "stacker.joblib").is_file():
        print("warn: meta-learner assente — fallback pesi statici 40/25/35")


if __name__ == "__main__":
    main()
