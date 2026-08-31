"""Training XGBoost binario (player A win) con OOF per calibrazione."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from modules.feature_engineering.features import FEATURE_COLS, FeatureEngineer

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "models" / "best_model.joblib"
OOF_PATH = ROOT / "data" / "models" / "oof_predictions.joblib"


class ModelTrainer:
    def __init__(self, feature_cols: list[str] | None = None):
        self.feature_cols = feature_cols or [c for c in FEATURE_COLS if c not in ("mkt_p_a", "mkt_overround")]
        self.model: XGBClassifier | None = None

    def train(self, features: pd.DataFrame | None = None, *, n_splits: int = 5) -> dict:
        if features is None:
            features = FeatureEngineer().build()

        df = features.dropna(subset=[c for c in self.feature_cols if c in features.columns]).copy()
        if "tourney_date" in df.columns:
            df = df.sort_values("tourney_date", kind="mergesort").reset_index(drop=True)
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].astype(float).fillna(0)
        y = df["label"].astype(int)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof = np.zeros(len(df))
        models: list[XGBClassifier] = []

        for fold, (tr, te) in enumerate(tscv.split(X)):
            m = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold,
                eval_metric="logloss",
            )
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
            models.append(m)

        self.model = models[-1]
        self.model.fit(X, y)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "feature_cols": cols,
            "n_train": len(df),
        }
        joblib.dump(bundle, MODEL_PATH)

        oof_df = df[["tourney_date", "winner_name", "loser_name", "surface"]].copy()
        oof_df["p_ml"] = oof
        oof_df["p_elo"] = df["e_w_elo"].values
        oof_df["elo_diff"] = df["elo_diff"].values
        oof_df["label"] = df["label"].values
        if "best_of" in df.columns:
            oof_df["best_of"] = df["best_of"].values
        if "odds_winner" in df.columns:
            oof_df["odds_winner"] = df["odds_winner"]
            oof_df["odds_loser"] = df["odds_loser"]
        joblib.dump(oof_df, OOF_PATH)

        from modules.model_training.stacker import train_stacker_from_features

        stack_info = train_stacker_from_features(df.assign(p_ml=oof))

        metrics = {
            "brier": float(brier_score_loss(y, oof)),
            "logloss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
            "n_train": len(df),
            "path": str(MODEL_PATH),
            "stacker": stack_info,
        }
        return metrics
