"""Pipeline orchestrata: sync dati → feature store → XGBoost + meta-learner."""

from __future__ import annotations

import json
import os
import sys
import time
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


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_full_ml_pipeline(*, min_year: int | None = None, force_features: bool | None = None) -> dict:
    from modules.data_update.tml import sync_tml
    from modules.data_update.sackmann import clone_sackmann_tour
    from modules.data_update.tennis_data_odds import download_tennis_data_odds
    from modules.dataset_loader import DatasetLoader
    from modules.feature_engineering.feature_store import FEATURES_PARQUET, build_feature_store
    from modules.model_training import ModelTrainer

    if min_year is None:
        min_year = int(os.environ.get("CLOUD_MIN_YEAR", "2010"))
    if force_features is None:
        force_features = os.environ.get("CLOUD_FORCE_FEATURES", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    t0 = time.perf_counter()

    _log("1/5 Sync TML (ATP primario)...")
    tml_info = sync_tml(clone=True, pull=True)
    _log(json.dumps(tml_info, ensure_ascii=False))

    _log("2/5 Sync Sackmann WTA fallback...")
    for tour in ("wta",):
        try:
            info = clone_sackmann_tour(tour=tour, dest=ROOT / "data" / "raw" / tour)
            _log(f"Sackmann {tour}: {json.dumps(info, ensure_ascii=False)}")
        except Exception as exc:
            _log(f"Sackmann {tour} skip: {exc}")

    _log("3/5 Odds tennis-data.co.uk...")
    odds_info = download_tennis_data_odds(force=False)
    _log(json.dumps(odds_info, ensure_ascii=False, default=str))

    _log(f"4/5 Build matches + feature store (min_year={min_year}, force_features={force_features})...")
    t_feat = time.perf_counter()
    loader = DatasetLoader(min_year=min_year)
    matches = loader.build()
    features = build_feature_store(force=force_features)
    _log(
        f"matches={len(matches)} features={len(features)} parquet={FEATURES_PARQUET} "
        f"elapsed={time.perf_counter() - t_feat:.0f}s"
    )

    _log("5/5 Training XGBoost + meta-learner...")
    t_train = time.perf_counter()
    metrics = ModelTrainer().train(features)
    _log(f"training elapsed={time.perf_counter() - t_train:.0f}s total={time.perf_counter() - t0:.0f}s")

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
