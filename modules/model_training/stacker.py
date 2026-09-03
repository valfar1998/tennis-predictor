"""Meta-learner (stacking) per blend Markov + Elo + ML — OOF temporale (TimeSeriesSplit)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from modules.markov.chain import estimate_serve_probs, match_win_prob

ROOT = Path(__file__).resolve().parents[2]
STACKER_PATH = ROOT / "data" / "models" / "stacker.joblib"
META_LEARNER_PATH = ROOT / "data" / "models" / "meta_learner.joblib"
CAL_PATH = ROOT / "data" / "models" / "calibration.json"
OOF_STACKER_PATH = ROOT / "data" / "models" / "stacker_oof.joblib"

META_COLS = ("p_markov", "p_elo", "p_ml")
UNCERTAINTY_COLS = ("mkt_divergence", "tourney_level_code", "data_density")
FALLBACK_WEIGHTS = {"p_ml": 0.35, "p_markov": 0.40, "p_elo": 0.25}
MIN_STACKER_ROWS = 500


def _stacker_artifact() -> Path | None:
    for p in (META_LEARNER_PATH, STACKER_PATH):
        if p.is_file():
            return p
    return None


def compute_p_markov_row(elo_diff: float, *, best_of: int = 3) -> float:
    p_a, p_b = estimate_serve_probs(1500 + elo_diff / 2, 1500 - elo_diff / 2)
    return match_win_prob(p_a, p_b, best_of=best_of)


def _sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinamento strict per data — prerequisito TimeSeriesSplit."""
    if "tourney_date" not in df.columns:
        return df.reset_index(drop=True)
    out = df.copy()
    out["_sort_date"] = pd.to_datetime(out["tourney_date"], errors="coerce")
    out = out.sort_values("_sort_date", kind="mergesort").drop(columns=["_sort_date"])
    return out.reset_index(drop=True)


def _prepare_meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = _sort_by_date(df)
    if "p_markov" not in work.columns and "elo_diff" in work.columns:
        work["p_markov"] = work.apply(
            lambda r: compute_p_markov_row(float(r["elo_diff"]), best_of=int(r.get("best_of") or 3)),
            axis=1,
        )
    elif "p_markov" not in work.columns and "e_w_elo" in work.columns:
        work["p_markov"] = work["e_w_elo"]
    if "p_elo" not in work.columns and "e_w_elo" in work.columns:
        work["p_elo"] = work["e_w_elo"]
    if "mkt_divergence" not in work.columns and "e_w_elo" in work.columns and "mkt_p_a" in work.columns:
        work["mkt_divergence"] = work.apply(
            lambda r: abs(float(r["e_w_elo"]) - float(r["mkt_p_a"]))
            if pd.notna(r.get("mkt_p_a"))
            else 0.0,
            axis=1,
        )
    if "tourney_level_code" not in work.columns and "tourney_level" in work.columns:
        from modules.constants import TOURNEY_LEVEL_CODE

        work["tourney_level_code"] = work["tourney_level"].map(
            lambda lv: TOURNEY_LEVEL_CODE.get(str(lv or "A").upper()[:1], 2.0)
        )
    return work


def _meta_columns(work: pd.DataFrame) -> list[str]:
    cols = [c for c in META_COLS if c in work.columns]
    for c in UNCERTAINTY_COLS:
        if c in work.columns:
            cols.append(c)
    return cols


def _uncertainty_defaults() -> dict[str, float]:
    return {
        "mkt_divergence": 0.0,
        "tourney_level_code": 2.0,
        "data_density": 50.0,
    }


class MetaStacker:
    def __init__(self):
        self.model: LogisticRegression | None = None
        self.weights: dict[str, float] = dict(FALLBACK_WEIGHTS)
        self._artifact: Path | None = None

    def _load_bundle(self) -> dict | None:
        path = _stacker_artifact()
        if not path:
            return None
        if self._artifact == path and self.model is not None:
            return {"model": self.model, "weights": self.weights}
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.weights = bundle.get("weights", FALLBACK_WEIGHTS)
        self._artifact = path
        return bundle

    def fit(self, df: pd.DataFrame, *, n_splits: int = 5) -> dict:
        work = _prepare_meta_frame(df)
        cols = _meta_columns(work)
        if len(cols) < 2:
            return {"ok": False, "error": "colonne meta insufficienti"}

        sub = work.dropna(subset=[c for c in META_COLS if c in cols] + ["label"]).copy()
        for c in UNCERTAINTY_COLS:
            if c in cols and c not in sub.columns:
                sub[c] = _uncertainty_defaults().get(c, 0.0)
        sub = sub.dropna(subset=cols)
        if len(sub) < MIN_STACKER_ROWS:
            return {"ok": False, "error": f"sample troppo piccolo ({len(sub)})"}

        X = sub[cols].astype(float).values
        y = sub["label"].astype(int).values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof = np.full(len(sub), np.nan)
        fold_models: list[LogisticRegression] = []

        for fold, (tr, te) in enumerate(tscv.split(X)):
            lr = LogisticRegression(max_iter=500, C=1.0, random_state=42 + fold)
            lr.fit(X[tr], y[tr])
            oof[te] = lr.predict_proba(X[te])[:, 1]
            fold_models.append(lr)

        mask = ~np.isnan(oof)
        brier_oof = float(brier_score_loss(y[mask], oof[mask])) if mask.sum() > 0 else None

        lr_final = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        lr_final.fit(X, y)
        self.model = lr_final

        coef = lr_final.coef_[0]
        raw = {c: float(w) for c, w in zip(cols, coef)}
        total = sum(abs(v) for v in raw.values()) or 1.0
        self.weights = {k: round(abs(v) / total, 4) for k, v in raw.items()}

        pred_in = lr_final.predict_proba(X)[:, 1]
        brier_in = float(brier_score_loss(y, pred_in))

        STACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": lr_final,
            "cols": cols,
            "weights": self.weights,
            "cv": "TimeSeriesSplit",
            "n_splits": n_splits,
            "brier_oof": brier_oof,
        }
        joblib.dump(bundle, STACKER_PATH)
        joblib.dump(bundle, META_LEARNER_PATH)

        oof_df = sub[["tourney_date", "winner_name", "loser_name"]].copy() if "winner_name" in sub.columns else pd.DataFrame()
        if not oof_df.empty:
            oof_df["p_stacker_oof"] = oof
            oof_df["label"] = y
            joblib.dump(oof_df, OOF_STACKER_PATH)

        # Calibrazione probabilità su OOF (Isotonic; fallback Platt se fallisce)
        cal_info: dict = {"ok": False}
        if mask.sum() >= 200:
            from modules.calibration.prob_calibrator import fit_calibrator_from_oof

            cal_info = fit_calibrator_from_oof(y[mask], oof[mask], method="isotonic")
            if not cal_info.get("ok"):
                cal_info = fit_calibrator_from_oof(y[mask], oof[mask], method="platt")

        cal = {}
        if CAL_PATH.is_file():
            try:
                cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        cal["stacker"] = {
            "weights": self.weights,
            "cols": cols,
            "brier": round(brier_oof or brier_in, 5),
            "brier_oof": round(brier_oof, 5) if brier_oof is not None else None,
            "brier_insample": round(brier_in, 5),
            "cv": "TimeSeriesSplit",
            "n_splits": n_splits,
            "n_train": len(sub),
        }
        if cal_info.get("ok"):
            cal["probability_calibration"] = {
                "method": cal_info.get("method"),
                "n_fit": cal_info.get("n_fit"),
                "brier_raw": cal_info.get("brier_raw"),
                "brier_cal": cal_info.get("brier_cal"),
                "path": cal_info.get("path"),
            }
        CAL_PATH.write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "ok": True,
            "brier": brier_oof or brier_in,
            "brier_oof": brier_oof,
            "brier_insample": brier_in,
            "weights": self.weights,
            "n_train": len(sub),
            "cv": "TimeSeriesSplit",
            "probability_calibration": cal_info if cal_info.get("ok") else None,
        }

    def predict(
        self,
        *,
        p_markov: float,
        p_elo: float,
        p_ml: float | None = None,
        mkt_divergence: float | None = None,
        tourney_level_code: float | None = None,
        data_density: float | None = None,
    ) -> float:
        if self.model is None:
            self._load_bundle()

        defaults = _uncertainty_defaults()
        extra = {
            "mkt_divergence": mkt_divergence if mkt_divergence is not None else defaults["mkt_divergence"],
            "tourney_level_code": tourney_level_code if tourney_level_code is not None else defaults["tourney_level_code"],
            "data_density": data_density if data_density is not None else defaults["data_density"],
        }

        if self.model is not None:
            cols = getattr(self.model, "feature_names_in_", None)
            vec = []
            mapping = {
                "p_markov": p_markov,
                "p_elo": p_elo,
                "p_ml": p_ml if p_ml is not None else p_elo,
                **extra,
            }
            if cols is not None:
                for c in cols:
                    vec.append(mapping.get(str(c), p_elo))
            else:
                bundle = self._load_bundle() or {}
                for c in bundle.get("cols", META_COLS):
                    vec.append(mapping.get(c, p_elo))
            return float(self.model.predict_proba(np.array([vec]))[0, 1])

        parts, weights = [], []
        if p_ml is not None:
            parts.append(p_ml)
            weights.append(self.weights.get("p_ml", FALLBACK_WEIGHTS["p_ml"]))
        parts.extend([p_markov, p_elo])
        weights.extend([
            self.weights.get("p_markov", FALLBACK_WEIGHTS["p_markov"]),
            self.weights.get("p_elo", FALLBACK_WEIGHTS["p_elo"]),
        ])
        wsum = sum(weights)
        return sum(p * w for p, w in zip(parts, weights)) / wsum


def train_stacker_from_features(features: pd.DataFrame | None = None, *, n_splits: int = 5) -> dict:
    from modules.feature_engineering.features import FeatureEngineer

    df = features if features is not None else FeatureEngineer().build()
    df = _sort_by_date(df)

    if "p_ml" not in df.columns:
        oof_path = ROOT / "data" / "models" / "oof_predictions.joblib"
        if oof_path.is_file():
            oof = joblib.load(oof_path)
            if "p_ml" in oof.columns:
                oof_sorted = _sort_by_date(oof)
                if len(oof_sorted) == len(df):
                    df = df.copy()
                    df["p_ml"] = oof_sorted["p_ml"].values
                else:
                    merged = df.merge(
                        oof_sorted[["tourney_date", "winner_name", "loser_name", "p_ml"]],
                        on=["tourney_date", "winner_name", "loser_name"],
                        how="left",
                    )
                    df = df.copy()
                    df["p_ml"] = merged["p_ml"].values
    if "p_ml" not in df.columns:
        df = df.copy()
        df["p_ml"] = df["e_w_elo"]

    df["p_elo"] = df["e_w_elo"]
    if "p_markov" not in df.columns:
        df["p_markov"] = df["elo_diff"].apply(lambda d: compute_p_markov_row(float(d)))
    return MetaStacker().fit(df, n_splits=n_splits)
