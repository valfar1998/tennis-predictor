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

from modules.ops_progress import OpProgress, log_done


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_full_ml_pipeline(*, min_year: int | None = None, force_features: bool | None = None) -> dict:
    from modules.advisor.validation_freeze import blocks_model_retrain, governance_status

    if blocks_model_retrain():
        status = governance_status()
        _log("VALIDATION FREEZE: retrain ML saltato — architettura congelata")
        _log(json.dumps(status, ensure_ascii=False, default=str))
        return {
            "ok": True,
            "skipped": "validation_freeze",
            "reason": "Nessun retrain durante finestra validazione 200-300 match BCR Pinnacle",
            "governance": status,
        }

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
    prog = OpProgress(5, label="retrain")

    prog.next("Sync TML (ATP primario)...")
    tml_info = sync_tml(clone=True, pull=True)
    _log(json.dumps(tml_info, ensure_ascii=False))

    prog.next("Sync Sackmann WTA fallback...")
    for tour in ("wta",):
        try:
            info = clone_sackmann_tour(tour=tour, dest=ROOT / "data" / "raw" / tour)
            _log(f"Sackmann {tour}: {json.dumps(info, ensure_ascii=False)}")
        except Exception as exc:
            _log(f"Sackmann {tour} skip: {exc}")

    prog.next("Odds tennis-data.co.uk...")
    odds_info = download_tennis_data_odds(force=False)
    _log(json.dumps(odds_info, ensure_ascii=False, default=str))

    prog.next(f"Build matches + feature store (min_year={min_year}, force_features={force_features})...")
    t_feat = time.perf_counter()
    loader = DatasetLoader(min_year=min_year)
    matches = loader.build()
    features = build_feature_store(force=force_features)
    _log(
        f"matches={len(matches)} features={len(features)} parquet={FEATURES_PARQUET} "
        f"elapsed={time.perf_counter() - t_feat:.0f}s"
    )

    prog.next("Training XGBoost + meta-learner...")
    t_train = time.perf_counter()
    metrics = ModelTrainer().train(features)
    _log(f"training elapsed={time.perf_counter() - t_train:.0f}s total={time.perf_counter() - t0:.0f}s")

    stacker = metrics.get("stacker") or {}
    brier = stacker.get("brier") or metrics.get("brier")
    log_done(f"Pipeline completata. Brier stacker/XGB: {brier}")

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
