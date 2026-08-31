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
SACKMANN_WTA_REPO = "https://github.com/JeffSackmann/tennis_wta.git"
ATP_DIR = ROOT / "data" / "raw" / "atp"
WTA_DIR = ROOT / "data" / "raw" / "wta"
MIN_YEAR = 2010


def _ensure_sackmann_tour(*, repo: str, dest: Path, players_file: str, match_glob: str) -> None:
    if (dest / players_file).is_file() and any(dest.glob(match_glob)):
        print(f"Sackmann {dest.name}: cache locale OK")
        return
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo, str(dest)],
        check=True,
    )
    print(f"Sackmann {dest.name}: clone completato")


def _ensure_sackmann() -> None:
    _ensure_sackmann_tour(
        repo=SACKMANN_REPO,
        dest=ATP_DIR,
        players_file="atp_players.csv",
        match_glob="atp_matches_*.csv",
    )
    try:
        _ensure_sackmann_tour(
            repo=SACKMANN_WTA_REPO,
            dest=WTA_DIR,
            players_file="wta_players.csv",
            match_glob="wta_matches_*.csv",
        )
    except subprocess.CalledProcessError as exc:
        print(f"Sackmann WTA clone fallito (opzionale): {exc}")


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
