from modules.calibration.config import load_calibration, save_calibration

__all__ = [
    "load_calibration",
    "save_calibration",
    "run_backtest",
    "apply_probability_calibration",
    "fit_calibrator_from_oof",
]


def run_backtest(*args, **kwargs):
    from modules.calibration.backtest import run_backtest as _run
    return _run(*args, **kwargs)


def apply_probability_calibration(*args, **kwargs):
    from modules.calibration.prob_calibrator import apply_probability_calibration as _fn
    return _fn(*args, **kwargs)


def fit_calibrator_from_oof(*args, **kwargs):
    from modules.calibration.prob_calibrator import fit_calibrator_from_oof as _fn
    return _fn(*args, **kwargs)
