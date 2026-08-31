"""Meta-learner (stacking) per blend Markov + Elo + ML."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from modules.feature_engineering.elo import expected_score
from modules.markov.chain import estimate_serve_probs, match_win_prob

ROOT = Path(__file__).resolve().parents[2]
STACKER_PATH = ROOT / "data" / "models" / "stacker.joblib"
CAL_PATH = ROOT / "data" / "models" / "calibration.json"

META_COLS = ("p_markov", "p_elo", "p_ml")


def compute_p_markov_row(elo_diff: float, *, best_of: int = 3) -> float:
    p_a, p_b = estimate_serve_probs(1500 + elo_diff / 2, 1500 - elo_diff / 2)
    return match_win_prob(p_a, p_b, best_of=best_of)


class MetaStacker:
    def __init__(self):
        self.model: LogisticRegression | None = None
        self.weights: dict[str, float] = {
            "p_ml": 0.35,
            "p_markov": 0.40,
            "p_elo": 0.25,
        }

    def fit(self, df: pd.DataFrame) -> dict:
        work = df.copy()
        if "p_markov" not in work.columns and "elo_diff" in work.columns:
            work["p_markov"] = work.apply(
                lambda r: compute_p_markov_row(float(r["elo_diff"]), best_of=int(r.get("best_of") or 3)),
                axis=1,
            )
        elif "p_markov" not in work.columns and "e_w_elo" in work.columns:
            work["p_markov"] = work["e_w_elo"]  # fallback grezzo

        cols = [c for c in META_COLS if c in work.columns]
        if len(cols) < 2:
            return {"ok": False, "error": "colonne meta insufficienti"}

        sub = work.dropna(subset=cols + ["label"]).copy()
        if len(sub) < 500:
            return {"ok": False, "error": f"sample troppo piccolo ({len(sub)})"}

        X = sub[cols].astype(float).values
        y = sub["label"].astype(int).values

        lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        lr.fit(X, y)
        self.model = lr

        coef = lr.coef_[0]
        raw = {c: float(w) for c, w in zip(cols, coef)}
        total = sum(abs(v) for v in raw.values()) or 1.0
        self.weights = {k: round(abs(v) / total, 4) for k, v in raw.items()}

        pred = lr.predict_proba(X)[:, 1]
        brier = float(brier_score_loss(y, pred))

        STACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": lr, "cols": cols, "weights": self.weights}, STACKER_PATH)

        cal = {}
        if CAL_PATH.is_file():
            try:
                cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        cal["stacker"] = {
            "weights": self.weights,
            "cols": cols,
            "brier": round(brier, 5),
            "n_train": len(sub),
        }
        CAL_PATH.write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")

        return {"ok": True, "brier": brier, "weights": self.weights, "n_train": len(sub)}

    def predict(self, *, p_markov: float, p_elo: float, p_ml: float | None = None) -> float:
        if STACKER_PATH.is_file() and self.model is None:
            bundle = joblib.load(STACKER_PATH)
            self.model = bundle["model"]
            self.weights = bundle.get("weights", self.weights)

        if self.model is not None:
            cols = getattr(self.model, "feature_names_in_", None)
            vec = []
            mapping = {"p_markov": p_markov, "p_elo": p_elo, "p_ml": p_ml if p_ml is not None else p_elo}
            if cols is not None:
                for c in cols:
                    vec.append(mapping.get(str(c), p_elo))
            else:
                bundle = joblib.load(STACKER_PATH) if STACKER_PATH.is_file() else {}
                for c in bundle.get("cols", META_COLS):
                    vec.append(mapping.get(c, p_elo))
            return float(self.model.predict_proba(np.array([vec]))[0, 1])

        parts, weights = [], []
        if p_ml is not None:
            parts.append(p_ml)
            weights.append(self.weights.get("p_ml", 0.35))
        parts.extend([p_markov, p_elo])
        weights.extend([self.weights.get("p_markov", 0.40), self.weights.get("p_elo", 0.25)])
        wsum = sum(weights)
        return sum(p * w for p, w in zip(parts, weights)) / wsum


def train_stacker_from_features(features: pd.DataFrame | None = None) -> dict:
    from modules.feature_engineering.features import FeatureEngineer

    df = features if features is not None else FeatureEngineer().build()
    if "p_ml" not in df.columns:
        oof_path = ROOT / "data" / "models" / "oof_predictions.joblib"
        if oof_path.is_file():
            oof = joblib.load(oof_path)
            if "p_ml" in oof.columns and len(oof) == len(df):
                df = df.copy()
                df["p_ml"] = oof["p_ml"].values
    if "p_ml" not in df.columns:
        df = df.copy()
        df["p_ml"] = df["e_w_elo"]

    df["p_elo"] = df["e_w_elo"]
    df["p_markov"] = df["elo_diff"].apply(lambda d: compute_p_markov_row(float(d)))
    return MetaStacker().fit(df)
