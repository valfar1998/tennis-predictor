"""Allena modello su GitHub Actions (Sackmann + tennis-data.co.uk)."""

from __future__ import annotations

import json
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SACKMANN_REPO = "https://github.com/JeffSackmann/tennis_atp.git"
ATP_DIR = ROOT / "data" / "raw" / "atp"
MIN_YEAR = 2010


def _ensure_sackmann() -> None:
    if (ATP_DIR / "atp_players.csv").is_file() and any(ATP_DIR.glob("atp_matches_*.csv")):
        print("Sackmann: cache locale OK")
        return
    if ATP_DIR.exists():
        shutil.rmtree(ATP_DIR, ignore_errors=True)
    ATP_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", SACKMANN_REPO, str(ATP_DIR)],
        check=True,
    )
    print("Sackmann: clone completato")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    _ensure_sackmann()

    from modules.data_update.tennis_data_odds import download_tennis_data_odds
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
