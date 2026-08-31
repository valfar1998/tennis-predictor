"""Feature store Parquet — cache veloce per retrain pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modules.dataset_loader.loader import MATCHES_PATH
from modules.feature_engineering.features import FEATURES_PATH, FeatureEngineer

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PARQUET = ROOT / "data" / "processed" / "features_v2.parquet"


def _needs_rebuild(*, force: bool, parquet: Path, deps: list[Path]) -> bool:
    if force or not parquet.is_file():
        return True
    p_mtime = parquet.stat().st_mtime
    for dep in deps:
        if dep.is_file() and dep.stat().st_mtime > p_mtime:
            return True
    return False


def build_feature_store(
    *,
    force: bool = False,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Genera feature store (CSV + Parquet)."""
    out = save_path or FEATURES_PARQUET
    deps = [MATCHES_PATH, ROOT / "data" / "processed" / "matches.csv"]
    if not _needs_rebuild(force=force, parquet=out, deps=deps):
        return pd.read_parquet(out)

    df = FeatureEngineer().build(save=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def load_feature_store(*, force_rebuild: bool = False) -> pd.DataFrame:
    """Carica feature store; ricostruisce se mancante o stale."""
    if force_rebuild:
        return build_feature_store(force=True)
    if FEATURES_PARQUET.is_file():
        if not _needs_rebuild(force=False, parquet=FEATURES_PARQUET, deps=[MATCHES_PATH]):
            return pd.read_parquet(FEATURES_PARQUET)
    if FEATURES_PATH.is_file():
        return pd.read_csv(FEATURES_PATH, low_memory=False, parse_dates=["tourney_date"])
    return build_feature_store()
