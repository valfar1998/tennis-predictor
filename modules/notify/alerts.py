"""Avvisi Telegram value bet tennis (stile football-predictor)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from modules.notify.telegram import load_credentials, send_message, telegram_status

from modules.advisor.online_learn import effective_alert_min_playability

ROOT = Path(__file__).resolve().parents[2]
SENT = ROOT / "data" / "processed" / "telegram_alerts_sent.json"
KEEP_DAYS = 21
CHUNK = 8
BRAND = "TENNIS_PREDICTOR"
TZ = ZoneInfo("Europe/Rome")


def _min_playability() -> float:
    return float(effective_alert_min_playability())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def brand_header(*, continued: bool = False) -> str:
    when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    suffix = " (cont.)" if continued else ""
    return f"{BRAND} — alert scommesse{suffix}\n{when} Roma"


def _load_sent() -> dict[str, str]:
    if not SENT.is_file():
        return {}
    try:
        raw = json.loads(SENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def _save_sent(ids: dict[str, str]) -> None:
    cutoff = _now() - timedelta(days=KEEP_DAYS)
    kept: dict[str, str] = {}
    for key, ts in ids.items():
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            when = _now()
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= cutoff:
            kept[key] = ts
    SENT.parent.mkdir(parents=True, exist_ok=True)
    SENT.write_text(json.dumps(kept, indent=2), encoding="utf-8")


def alert_key(pred: dict) -> str:
    return "|".join(
        str(pred.get(k) or "")
        for k in ("player_a", "player_b", "date", "tourney", "betfair_event_id")
    )


def _format_bet(pred: dict) -> str:
    rec = pred.get("recommended") or {}
    date = str(pred.get("date") or "")[:10]
    tourney = str(pred.get("tourney") or "").strip()
    surface = str(pred.get("surface") or "")
    source = str(pred.get("odds_source") or "book")
    player_a = str(pred.get("player_a") or "?")
    player_b = str(pred.get("player_b") or "?")
    pick = str(rec.get("player") or "?")
    odds = rec.get("odds")
    p = rec.get("probability", 0)
    ev = rec.get("ev", 0)
    ev_pct = rec.get("ev_pct", float(ev) * 100)
    kelly = rec.get("kelly", 0)
    sig = pred.get("market_signals") or {}
    parts = pred.get("playability_parts") or {}

    meta = " · ".join(x for x in (date, tourney) if x)
    lines = [
        meta or date,
        f"{player_a} vs {player_b}",
        f"Pick: {pick} @ {odds}",
        f"P={float(p):.1%} | EV={float(ev_pct):+.1f}% | Kelly={float(kelly):.2%}",
    ]
    # Quote suggerite: fair modello + minimo per tenere edge
    try:
        p_f = float(p or 0)
        if p_f > 0.01:
            fair = 1.0 / p_f
            from modules.advisor.online_learn import effective_min_edge

            min_edge = float(effective_min_edge())
            min_odds = (1.0 + min_edge) / p_f
            lines.append(
                f"Quote suggerite: fair {fair:.2f} | min @{min_odds:.2f} (edge>={min_edge:.0%})"
            )
    except Exception:
        pass
    lines.append(
        f"Giocabilità: {int(pred.get('playability') or 0)}/100 ({pred.get('playability_label') or '—'})"
    )
    analysis = pred.get("analysis") or {}
    if analysis.get("telegram_line"):
        lines.append(f"Analisi: {analysis['telegram_line']}")
    elif analysis.get("p_signal") is not None:
        fair = analysis.get("fair_odds")
        fair_s = f" | fair {float(fair):.2f}" if fair else ""
        lines.append(
            f"Analisi: signal {float(analysis['p_signal']):.0%} "
            f"accordo {float(analysis.get('consensus_agree') or 0):.0%}{fair_s}"
        )
    mw = sig.get("volume_pct_pick")
    drop = sig.get("drop_pct")
    mw_missing = bool(sig.get("missing")) and mw is None
    drop_missing = drop is None and (sig.get("source") == "oddssafari" or parts.get("signals_missing"))
    sig_bits = []
    if mw is not None:
        sig_bits.append(f"Moneyway {float(mw):.0f}% vol")
    elif mw_missing or parts.get("signals_missing"):
        sig_bits.append("Moneyway n/d")
    if drop is not None:
        tag = "allineato" if sig.get("aligned_with_pick") else "contro pick"
        sig_bits.append(f"Drop {float(drop):.0f}% ({tag})")
    elif drop_missing or parts.get("signals_missing"):
        sig_bits.append("Drop n/d")
    if sig_bits:
        lines.append("Segnali: " + " · ".join(sig_bits))
    if parts.get("moneyway") is not None:
        mw_s = parts["moneyway"]
        drop_s = parts.get("dropping_odds", 0)
        note = " (segnali parziali/assenti)" if parts.get("signals_missing") else ""
        lines.append(f"Score MW={mw_s:.2f} Drop={drop_s:.2f}{note}")
    if rec.get("odds_sharpe") is not None:
        lines.append(
            f"Sharpe-like={float(rec['odds_sharpe']):.3f} | "
            f"KellyAdj={float(rec.get('kelly_adj_rank') or 0):.4f}"
        )
    lines.append(f"Fonte quote: {source} | Superficie: {surface}")
    news = pred.get("news_alert") or {}
    if news.get("any_alert"):
        cat = news.get("category") or "news"
        head = news.get("headline") or ""
        flag = "HARD BLOCK" if news.get("hard_block_pick") or pred.get("news_hard_block") else "ALERT"
        lines.append(f"News {flag} ({cat}): {head[:120]}")
    if rec.get("retirement_warning"):
        lines.append(str(rec["retirement_warning"]))
    elif pred.get("p_retire"):
        lines.append(f"P(ritiro)≈{float(pred['p_retire']):.0%}")
    return "\n".join(lines)


def _pack(title: str, items: list[dict]) -> list[tuple[str, list[str]]]:
    if not items:
        return []
    out: list[tuple[str, list[str]]] = []
    for i in range(0, len(items), CHUNK):
        chunk = items[i : i + CHUNK]
        head = title if i == 0 else f"{title} (cont.)"
        body = "\n\n".join(_format_bet(p) for p in chunk)
        msg = f"{brand_header(continued=i > 0)}\n\n{head}\n\n{body}"
        out.append((msg, [alert_key(p) for p in chunk]))
    return out


def dispatch_alerts(predictions: list[dict] | None = None, *, dry_run: bool = False) -> dict:
    """Invia value bet nuovi + alert news su bet/shadow (dedup)."""
    rows = predictions or []
    bets = [
        p for p in rows
        if p.get("action") == "bet"
        and p.get("recommended")
        and float(p.get("playability") or 0) >= _min_playability()
    ]
    sent_ids = _load_sent()
    fresh = [p for p in bets if alert_key(p) not in sent_ids]
    messages = _pack(f"GIOCA · giocabilità ≥{int(_min_playability())}", fresh)

    # News: withdrawal/injury su bet/shadow/review o hard-block
    news_items = []
    for p in rows:
        na = p.get("news_alert") or {}
        if not na.get("any_alert"):
            continue
        interesting = (
            p.get("action") in ("bet", "shadow", "review")
            or p.get("news_hard_block")
            or na.get("hard_block_pick")
            or float((na.get("pick") or {}).get("severity") or 0) >= 0.70
            or float((na.get("opponent") or {}).get("severity") or 0) >= 0.85
        )
        if not interesting:
            continue
        key = "news|" + alert_key(p)
        if key in sent_ids:
            continue
        news_items.append(p)
    news_messages = _pack("NEWS / INJURY · attenzione palinsesto", news_items)

    sent_n = 0
    if dry_run:
        for msg, _ids in messages + news_messages:
            print(msg)
            print("---")
    else:
        if (messages or news_messages) and not load_credentials():
            print("telegram skip: credenziali assenti")
        elif messages or news_messages:
            now = _now().isoformat()
            changed = False
            for msg, ids in messages:
                if send_message(msg):
                    sent_n += 1
                    for key in ids:
                        sent_ids[key] = now
                    changed = True
                    try:
                        from modules.advisor.slippage_audit import log_alert

                        id_set = set(ids)
                        for p in fresh:
                            if alert_key(p) in id_set:
                                log_alert(p, sent_at=now)
                    except Exception:
                        pass
            for msg, ids in news_messages:
                if send_message(msg):
                    sent_n += 1
                    for key in ids:
                        sent_ids["news|" + key] = now
                    changed = True
            if changed:
                _save_sent(sent_ids)

    info = {
        "n_bets": len(bets),
        "n_new_bets": len(fresh),
        "n_news_alerts": len(news_items),
        "n_messages": len(messages) + len(news_messages),
        "n_sent": sent_n,
        "dry_run": dry_run,
        "status": telegram_status(),
    }
    print(
        f"telegram avvisi: value {info['n_new_bets']}/{info['n_bets']} nuovi, "
        f"news {info['n_news_alerts']}, inviati {sent_n}"
    )
    return info