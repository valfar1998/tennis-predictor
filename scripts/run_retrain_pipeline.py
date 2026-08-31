"""Pipeline orchestrata: sync dati → feature store → XGBoost + meta-learner."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings(
    "ignore",
    message="Unknown extension is not supported and will be removed",
    category=UserWarning,
)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def run_full_ml_pipeline(*, min_year: int = 2010, force_features: bool = False) -> dict:
    from modules.data_update.tml import sync_tml
    from modules.data_update.sackmann import clone_sackmann_tour
    from modules.data_update.tennis_data_odds import download_tennis_data_odds
    from modules.dataset_loader import DatasetLoader
    from modules.feature_engineering.feature_store import FEATURES_PARQUET, build_feature_store
    from modules.model_training import ModelTrainer

    print("1/5 Sync TML (ATP primario)...")
    tml_info = sync_tml(clone=True, pull=True)
    print(json.dumps(tml_info, ensure_ascii=False))

    print("2/5 Sync Sackmann WTA fallback...")
    for tour in ("wta",):
        try:
            info = clone_sackmann_tour(tour=tour, dest=ROOT / "data" / "raw" / tour)
            print(f"Sackmann {tour}:", json.dumps(info, ensure_ascii=False))
        except Exception as exc:
            print(f"Sackmann {tour} skip: {exc}")

    print("3/5 Odds tennis-data.co.uk...")
    odds_info = download_tennis_data_odds(force=False)
    print(json.dumps(odds_info, ensure_ascii=False, default=str))

    print("4/5 Build matches + feature store Parquet...")
    loader = DatasetLoader(min_year=min_year)
    matches = loader.build()
    features = build_feature_store(force=force_features)
    print(f"matches={len(matches)} features={len(features)} parquet={FEATURES_PARQUET}")

    print("5/5 Training XGBoost + meta-learner...")
    metrics = ModelTrainer().train(features)

    stacker = metrics.get("stacker") or {}
    brier = stacker.get("brier") or metrics.get("brier")
    print(f"Pipeline completata. Brier stacker/XGB: {brier}")

    return {
        "ok": True,
        "min_year": min_year,
        "n_matches": int(len(matches)),
        "n_features": int(len(features)),
        "metrics": metrics,
        "tml": tml_info,
        "odds": odds_info,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    result = run_full_ml_pipeline()
    print(json.dumps(result, indent=2, default=str))
