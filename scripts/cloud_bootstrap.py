"""Allena modello su GitHub Actions (Sackmann + tennis-data.co.uk)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ATP_DIR = ROOT / "data" / "raw" / "atp"
WTA_DIR = ROOT / "data" / "raw" / "wta"
MIN_YEAR = 2010


def _ensure_sackmann() -> None:
    from modules.data_update.sackmann import clone_sackmann_tour

    for tour, dest in (("atp", ATP_DIR), ("wta", WTA_DIR)):
        info = clone_sackmann_tour(tour=tour, dest=dest)
        print(f"Sackmann {tour.upper()}:", json.dumps(info, ensure_ascii=False))
        if tour == "atp" and not info.get("ok"):
            raise SystemExit(f"bootstrap: mancano dati ATP — {info.get('error')}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    _ensure_sackmann()

    from modules.data_update.tennis_abstract import fetch_all_tennis_abstract_elo
    from modules.data_update.tennis_data_odds import download_tennis_data_odds

    fetch_all_tennis_abstract_elo(force=True)
    from modules.dataset_loader import DatasetLoader
    from modules.feature_engineering import FeatureEngineer
    from modules.model_training import ModelTrainer

    odds_info = download_tennis_data_odds(force=True)
    loader = DatasetLoader(min_year=MIN_YEAR)
    matches = loader.build()
    engineer = FeatureEngineer()
    features = engineer.build()
    train_info = ModelTrainer().train(features)

    info = {
        "cloud": True,
        "min_year": MIN_YEAR,
        "odds": odds_info,
        "n_matches": int(len(matches)),
        "n_features": int(len(features)),
        **{k: v for k, v in train_info.items() if k != "model"},
    }
    print(json.dumps(info, indent=2, default=str))

    model = ROOT / "data" / "models" / "best_model.joblib"
    if not model.is_file():
        raise SystemExit("bootstrap: manca data/models/best_model.joblib")


if __name__ == "__main__":
    main()
