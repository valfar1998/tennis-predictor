"""Calibrazione probabilità (Isotonic / Platt) su predizioni OOF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CALIBRATOR_PATH = ROOT / "data" / "models" / "prob_calibrator.joblib"

Method = Literal["isotonic", "platt"]


class ProbCalibrator:
    """Mappa P grezza → P calibrata (isotonic o sigmoid/Platt)."""

    def __init__(self, method: Method = "isotonic"):
        self.method: Method = method
        self.model: Any = None
        self.n_fit: int = 0
        self.brier_raw: float | None = None
        self.brier_cal: float | None = None

    def fit(self, y_true, p_raw, *, method: Method | None = None) -> dict:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss

        y = np.asarray(y_true, dtype=int).ravel()
        p = np.clip(np.asarray(p_raw, dtype=float).ravel(), 1e-6, 1 - 1e-6)
        if len(y) < 200 or len(np.unique(y)) < 2:
            return {"ok": False, "error": f"sample insufficiente ({len(y)})"}

        self.method = method or self.method
        self.brier_raw = float(brier_score_loss(y, p))

        if self.method == "platt":
            # Platt: logistic su logit(p)
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
            lr.fit(logit, y)
            self.model = {"type": "platt", "lr": lr}
            p_cal = lr.predict_proba(logit)[:, 1]
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            iso.fit(p, y)
            self.model = {"type": "isotonic", "iso": iso}
            p_cal = iso.predict(p)

        self.brier_cal = float(brier_score_loss(y, p_cal))
        self.n_fit = int(len(y))
        return {
            "ok": True,
            "method": self.method,
            "n_fit": self.n_fit,
            "brier_raw": round(self.brier_raw, 5),
            "brier_cal": round(self.brier_cal, 5),
        }

    def transform(self, p: float) -> float:
        if self.model is None:
            return float(np.clip(p, 0.01, 0.99))
        x = float(np.clip(p, 1e-6, 1 - 1e-6))
        kind = self.model.get("type")
        if kind == "platt":
            logit = np.log(x / (1 - x)).reshape(1, -1)
            return float(self.model["lr"].predict_proba(logit)[0, 1])
        return float(self.model["iso"].predict([x])[0])

    def save(self, path: Path | None = None) -> Path:
        path = path or CALIBRATOR_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "method": self.method,
                "model": self.model,
                "n_fit": self.n_fit,
                "brier_raw": self.brier_raw,
                "brier_cal": self.brier_cal,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ProbCalibrator | None":
        path = path or CALIBRATOR_PATH
        if not path.is_file():
            return None
        try:
            bundle = joblib.load(path)
        except Exception:
            return None
        obj = cls(method=bundle.get("method", "isotonic"))
        obj.model = bundle.get("model")
        obj.n_fit = int(bundle.get("n_fit") or 0)
        obj.brier_raw = bundle.get("brier_raw")
        obj.brier_cal = bundle.get("brier_cal")
        return obj if obj.model is not None else None


_CACHE: ProbCalibrator | None = None
_CACHE_LOADED = False


def get_calibrator(*, force: bool = False) -> ProbCalibrator | None:
    global _CACHE, _CACHE_LOADED
    if force or not _CACHE_LOADED:
        _CACHE = ProbCalibrator.load()
        _CACHE_LOADED = True
    return _CACHE


def apply_probability_calibration(p: float) -> tuple[float, dict]:
    """Applica calibratore se presente; altrimenti identity."""
    cal = get_calibrator()
    p0 = float(np.clip(p, 0.01, 0.99))
    if cal is None:
        return p0, {"applied": False, "method": None}
    p1 = float(np.clip(cal.transform(p0), 0.01, 0.99))
    return p1, {
        "applied": True,
        "method": cal.method,
        "p_raw": round(p0, 4),
        "p_cal": round(p1, 4),
        "n_fit": cal.n_fit,
        "brier_raw": cal.brier_raw,
        "brier_cal": cal.brier_cal,
    }


def fit_calibrator_from_oof(
    y_true,
    p_raw,
    *,
    method: Method = "isotonic",
) -> dict:
    """Fit + persistenza; aggiorna cache in-process."""
    global _CACHE, _CACHE_LOADED
    cal = ProbCalibrator(method=method)
    info = cal.fit(y_true, p_raw, method=method)
    if not info.get("ok"):
        return info
    path = cal.save()
    _CACHE = cal
    _CACHE_LOADED = True
    info["path"] = str(path)
    return info
