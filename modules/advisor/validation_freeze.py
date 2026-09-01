"""Finestra validazione live: architettura congelata, KPI unico = BCR Pinnacle."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "processed" / "validation_freeze.json"

DEFAULT_STATE = {
    "active": True,
    "started_at": "2026-09-01",
    "target_n": 250,
    "min_n": 200,
    "max_n": 300,
    "bcr_target": 0.55,
    "pinnacle_only": True,
    "policy": (
        "Nessuna modifica strutturale a pesi, feature o retrain ML fino al completamento "
        "della finestra. Solo settle + metriche BCR."
    ),
}


def _env_override() -> bool | None:
    raw = os.environ.get("LIVE_VALIDATION_FREEZE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return None


def load_state() -> dict[str, Any]:
    env = _env_override()
    if STATE_PATH.is_file():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            state = dict(DEFAULT_STATE)
    else:
        state = dict(DEFAULT_STATE)
    if env is not None:
        state["active"] = env
    state.setdefault("active", True)
    return state


def save_state(state: dict[str, Any]) -> Path:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return STATE_PATH


def is_frozen() -> bool:
    return bool(load_state().get("active", True))


def blocks_online_learn_writes() -> bool:
    return is_frozen()


def blocks_model_retrain() -> bool:
    return is_frozen()


def blocks_playability_learned_adjustments() -> bool:
    return is_frozen()


def validation_progress() -> dict[str, Any]:
    """Avanzamento finestra vs target BCR Pinnacle."""
    from modules.advisor.live_metrics import compute_bcr

    state = load_state()
    bcr = compute_bcr(pinnacle_only=bool(state.get("pinnacle_only", True)))
    n = int(bcr.get("n") or 0)
    min_n = int(state.get("min_n") or 200)
    max_n = int(state.get("max_n") or 300)
    target = int(state.get("target_n") or 250)

    return {
        "frozen": is_frozen(),
        "started_at": state.get("started_at"),
        "n_pinnacle_settled": n,
        "target_n": target,
        "min_n": min_n,
        "max_n": max_n,
        "window_complete": n >= min_n,
        "window_pct": round(min(100.0, 100.0 * n / target), 1) if target else None,
        "bcr_target": float(state.get("bcr_target") or 0.55),
        "bcr_current": bcr.get("bcr"),
        "bcr_pass": bcr.get("pass"),
        "policy": state.get("policy"),
    }


def governance_status() -> dict[str, Any]:
    state = load_state()
    progress = validation_progress()
    return {
        "validation_freeze": {
            **state,
            "active": is_frozen(),
            "blocks": {
                "online_learn_writes": blocks_online_learn_writes(),
                "model_retrain": blocks_model_retrain(),
                "playability_learned_adjustments": blocks_playability_learned_adjustments(),
            },
        },
        "progress": progress,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def format_freeze_banner() -> str:
    if not is_frozen():
        return "Validazione: FREEZE disattivo — online learn e retrain consentiti"
    p = validation_progress()
    bcr_s = f"{p['bcr_current']:.1%}" if p.get("bcr_current") is not None else "n/d"
    return (
        f"VALIDAZIONE LIVE (FREEZE): {p['n_pinnacle_settled']}/{p['target_n']} pick Pinnacle settle "
        f"| BCR {bcr_s} (target >{p['bcr_target']:.0%}) "
        f"| nessun aggiornamento pesi/feature fino a {p['min_n']}+ match"
    )
