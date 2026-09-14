"""Consenso analisi: forma / superficie / qualità / prior TA / stack → p_signal."""

from __future__ import annotations

from typing import Any

from modules.feature_engineering.elo import expected_score

# Pesi fissi (no online-learn finché non c'è sample sufficiente)
W_SURFACE = 0.30
W_FORM = 0.25
W_QUALITY = 0.20
W_EXTERNAL = 0.15
W_STACK = 0.10


def _clip(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, float(x)))


def _form_prob(form_a: float, form_b: float) -> float:
    """Win-rate last-N → P(A) via delta centrato su 0.5."""
    delta = float(form_a) - float(form_b)
    # scala: Δ0.20 WR ≈ +10pp
    return _clip(0.5 + delta * 0.50)


def _pillar_marks(parts: dict[str, float], pick_side: str) -> str:
    """Sintesi tipo Form+ Surf+ TA~ per Telegram."""
    want_high = pick_side == "A"
    bits: list[str] = []
    labels = (
        ("Form", "p_form"),
        ("Surf", "p_surface"),
        ("Qual", "p_quality"),
        ("TA", "p_external"),
        ("Stack", "p_stack"),
    )
    for label, key in labels:
        p = float(parts.get(key) or 0.5)
        lean_a = p >= 0.52
        lean_b = p <= 0.48
        if want_high:
            mark = "+" if lean_a else ("-" if lean_b else "~")
        else:
            mark = "+" if lean_b else ("-" if lean_a else "~")
        bits.append(f"{label}{mark}")
    return " ".join(bits)


def consensus_agree(pillars: list[float]) -> float:
    """1.0 se i pilastri concordano sul lato; scende con dispersione."""
    vals = [float(p) for p in pillars if p is not None]
    if len(vals) < 2:
        return 0.5
    mean = sum(vals) / len(vals)
    # Varianza intorno alla media; 0.08 std ≈ accordo basso
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var**0.5
    # Accordo sul lato: frazione che punta stesso verso di mean
    if mean >= 0.5:
        same = sum(1 for v in vals if v >= 0.48) / len(vals)
    else:
        same = sum(1 for v in vals if v <= 0.52) / len(vals)
    agree = 0.55 * same + 0.45 * max(0.0, 1.0 - std / 0.10)
    return max(0.0, min(1.0, agree))


def build_signal_consensus(
    prediction: dict[str, Any],
    *,
    features: dict[str, Any] | None = None,
    elo_surface_a: float | None = None,
    elo_surface_b: float | None = None,
    elo_global_a: float | None = None,
    elo_global_b: float | None = None,
    ta_elo_a: float | None = None,
    ta_elo_b: float | None = None,
) -> dict[str, Any]:
    """Calcola pilastri e ``p_signal``; aggiorna ``p_win_a_raw`` sulla predizione.

    Ritorna dict analisi; il caller fa merge su prediction.
    """
    feat = features or {}
    p_stack = float(
        prediction.get("p_win_a_stacker")
        or prediction.get("p_win_a")
        or 0.5
    )

    # Superficie
    if elo_surface_a is not None and elo_surface_b is not None:
        p_surface = expected_score(float(elo_surface_a), float(elo_surface_b))
        has_surface = True
    else:
        p_surface = float(prediction.get("p_elo") or p_stack)
        has_surface = prediction.get("p_elo") is not None

    # Qualità (Elo overall)
    if elo_global_a is not None and elo_global_b is not None:
        p_quality = expected_score(float(elo_global_a), float(elo_global_b))
        has_quality = True
    else:
        p_quality = p_surface
        has_quality = has_surface

    # Forma
    fa = feat.get("form_wr_10_a")
    fb = feat.get("form_wr_10_b")
    if fa is None:
        fa = feat.get("surface_wr_a", 0.5)
    if fb is None:
        fb = feat.get("surface_wr_b", 0.5)
    p_form = _form_prob(float(fa or 0.5), float(fb or 0.5))
    has_form = fa is not None and fb is not None

    # Prior esterno Tennis Abstract
    if ta_elo_a is not None and ta_elo_b is not None and float(ta_elo_a) > 0 and float(ta_elo_b) > 0:
        p_external = expected_score(float(ta_elo_a), float(ta_elo_b))
        has_external = True
    else:
        p_external = p_stack
        has_external = False

    weights = {
        "surface": (W_SURFACE, p_surface, has_surface),
        "form": (W_FORM, p_form, has_form),
        "quality": (W_QUALITY, p_quality, has_quality),
        "external": (W_EXTERNAL, p_external, has_external),
        "stack": (W_STACK, p_stack, True),
    }
    active = [(w, p) for w, p, ok in weights.values() if ok]
    if not active:
        p_signal = p_stack
    else:
        w_sum = sum(w for w, _ in active)
        p_signal = sum(w * p for w, p in active) / max(w_sum, 1e-9)

    p_signal = _clip(p_signal)
    pillars = [p_surface, p_form, p_quality]
    if has_external:
        pillars.append(p_external)
    pillars.append(p_stack)
    agree = consensus_agree(pillars)

    fair_odds_a = round(1.0 / p_signal, 3) if p_signal > 0.01 else None
    fair_odds_b = round(1.0 / (1.0 - p_signal), 3) if p_signal < 0.99 else None

    analysis = {
        "p_form": round(p_form, 4),
        "p_surface": round(p_surface, 4),
        "p_quality": round(p_quality, 4),
        "p_external": round(p_external, 4),
        "p_stack": round(p_stack, 4),
        "p_signal": round(p_signal, 4),
        "consensus_agree": round(agree, 4),
        "fair_odds_a": fair_odds_a,
        "fair_odds_b": fair_odds_b,
        "fair_odds": fair_odds_a,
        "weights": {
            "surface": W_SURFACE,
            "form": W_FORM,
            "quality": W_QUALITY,
            "external": W_EXTERNAL,
            "stack": W_STACK,
        },
        "pillars_present": {
            "surface": has_surface,
            "form": has_form,
            "quality": has_quality,
            "external": has_external,
            "stack": True,
        },
    }
    return analysis


def apply_signal_consensus(
    prediction: dict[str, Any],
    *,
    features: dict[str, Any] | None = None,
    elo_surface_a: float | None = None,
    elo_surface_b: float | None = None,
    elo_global_a: float | None = None,
    elo_global_b: float | None = None,
    ta_elo_a: float | None = None,
    ta_elo_b: float | None = None,
) -> dict[str, Any]:
    """Applica consenso: ``p_win_a_raw = p_signal`` per lo shrink Bayesiano."""
    out = dict(prediction)
    analysis = build_signal_consensus(
        out,
        features=features,
        elo_surface_a=elo_surface_a,
        elo_surface_b=elo_surface_b,
        elo_global_a=elo_global_a,
        elo_global_b=elo_global_b,
        ta_elo_a=ta_elo_a,
        ta_elo_b=ta_elo_b,
    )
    out["analysis"] = analysis
    out["consensus_agree"] = analysis["consensus_agree"]
    # Conserva stacker calibrato; lo shrink userà p_win_a_raw
    if out.get("p_win_a_stacker") is None and out.get("p_win_a") is not None:
        out["p_win_a_stacker"] = out["p_win_a"]
    out["p_win_a_raw"] = analysis["p_signal"]
    out["p_win_a"] = analysis["p_signal"]
    return out


def refresh_fair_odds_after_shrink(prediction: dict[str, Any]) -> dict[str, Any]:
    """Aggiorna fair odds sull'analisi usando P finale (post-mercato)."""
    out = dict(prediction)
    analysis = dict(out.get("analysis") or {})
    p = float(out.get("p_win_a") or analysis.get("p_signal") or 0.5)
    if p > 0.01:
        analysis["fair_odds_a"] = round(1.0 / p, 3)
        analysis["fair_odds"] = analysis["fair_odds_a"]
    if p < 0.99:
        analysis["fair_odds_b"] = round(1.0 / (1.0 - p), 3)
    pick_side = "A"
    rec = out.get("recommended") or out.get("best_play") or {}
    if rec.get("side") == "B":
        pick_side = "B"
        analysis["fair_odds"] = analysis.get("fair_odds_b")
    analysis["telegram_line"] = _pillar_marks(
        {
            "p_form": analysis.get("p_form", 0.5),
            "p_surface": analysis.get("p_surface", 0.5),
            "p_quality": analysis.get("p_quality", 0.5),
            "p_external": analysis.get("p_external", 0.5),
            "p_stack": analysis.get("p_stack", 0.5),
        },
        pick_side,
    )
    if analysis.get("fair_odds"):
        analysis["telegram_line"] += f" | fair {float(analysis['fair_odds']):.2f}"
    out["analysis"] = analysis
    return out
