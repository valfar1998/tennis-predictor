"""Avvisi Telegram value bet tennis (stile football-predictor)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from modules.notify.telegram import load_credentials, send_message, telegram_status

from modules.constants import MAX_ODDS_PLAY, MIN_EDGE, MIN_KELLY, MIN_ODDS_PLAY

ROOT = Path(__file__).resolve().parents[2]
SENT = ROOT / "data" / "processed" / "telegram_alerts_sent.json"
KEEP_DAYS = 21
CHUNK = 8
BRAND = "TENNIS_PREDICTOR"
TZ = ZoneInfo("Europe/Rome")


def _passes_value_filters(pred: dict) -> bool:
    """Filtri minimi allineati a bet: quota MIN–MAX, EV≥MIN_EDGE, Kelly≥MIN_KELLY.

    La qualità dell'edge dipende dallo shrink Bayesiano (P calibrata), non da
    un tetto artificiale sulle quote lunghe.
    """
    rec = pred.get("recommended") or {}
    try:
        odds = float(rec.get("odds") or 0)
        ev = float(rec.get("ev") or 0)
        # Dopo CB scale usa pre-scale se presente (sizing ≠ eleggibilità alert)
        kelly = float(rec.get("kelly_pre_cb_scale") or rec.get("kelly") or 0)
    except (TypeError, ValueError):
        return False
    if odds < MIN_ODDS_PLAY or odds > MAX_ODDS_PLAY:
        return False
    if ev < MIN_EDGE:
        return False
    if kelly < MIN_KELLY:
        return False
    return True


def _is_telegram_bet(pred: dict) -> bool:
    """Fase 1: action=bet + filtri core EV/Kelly/quota → alert (playability non silenzia)."""
    if pred.get("action") != "bet" or not pred.get("recommended"):
        return False
    return _passes_value_filters(pred)

def _now() -> datetime:
    return datetime.now(timezone.utc)


def brand_header(*, continued: bool = False) -> str:
    when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    suffix = " (cont.)" if continued else ""
    return f"{BRAND} — cosa fare{suffix}\n{when} Roma"


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
    """Messaggio essenziale: cosa puntare, a che quota, quanto (% Kelly)."""
    rec = pred.get("recommended") or {}
    date = str(pred.get("date") or "")[:10]
    tourney = str(pred.get("tourney") or "").strip()
    start = str(pred.get("start_time_local") or pred.get("commence_time_utc") or "").strip()
    player_a = str(pred.get("player_a") or "?")
    player_b = str(pred.get("player_b") or "?")
    pick = str(rec.get("player") or "?")
    source = str(pred.get("odds_source") or "book")
    try:
        odds = float(rec.get("odds") or 0)
    except (TypeError, ValueError):
        odds = 0.0
    try:
        kelly = float(rec.get("kelly") or 0)
    except (TypeError, ValueError):
        kelly = 0.0
    try:
        ev_pct = float(rec.get("ev_pct") if rec.get("ev_pct") is not None else float(rec.get("ev") or 0) * 100)
    except (TypeError, ValueError):
        ev_pct = 0.0

    when_bits = [x for x in (date, start[:16] if start else "", tourney) if x]
    lines = [
        f"PUNTA: {pick}",
        f"Quota: {odds:.2f}  |  Stake: {kelly:.2%} del bankroll",
        f"Match: {player_a} vs {player_b}",
    ]
    if when_bits:
        lines.append(" · ".join(when_bits))
    lines.append(f"EV {ev_pct:+.1f}% · fonte {source}")

    # Solo se cambia qualcosa di concreto per l'azione
    try:
        p_f = float(rec.get("probability") or 0)
        if p_f > 0.01:
            from modules.advisor.online_learn import effective_min_edge

            min_edge = float(effective_min_edge())
            min_odds = (1.0 + min_edge) / p_f
            if odds > 0 and odds < min_odds:
                lines.append(f"Attenzione: sotto min @{min_odds:.2f} (edge {min_edge:.0%}) → skip")
    except Exception:
        pass

    news = pred.get("news_alert") or {}
    if news.get("any_alert"):
        flag = "STOP" if news.get("hard_block_pick") or pred.get("news_hard_block") else "NEWS"
        head = str(news.get("headline") or "")[:100]
        lines.append(f"{flag}: {head}" if head else flag)
    if rec.get("retirement_warning"):
        lines.append(str(rec["retirement_warning"]))

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
    bets = [p for p in rows if _is_telegram_bet(p)]
    sent_ids = _load_sent()
    fresh = [p for p in bets if alert_key(p) not in sent_ids]
    messages = _pack(
        f"GIOCA ({len(fresh)}) — segui quota e stake Kelly sotto",
        fresh,
    )

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

def _avg_clv_recent(*, days: int = 14) -> dict:
    from datetime import date, timedelta

    from modules.data_update.history import load_history

    cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
    vals: list[float] = []
    n_bet = n_shadow = 0
    stake = 0.0
    for r in load_history(limit=3000):
        day = str(r.get("date") or "")[:10]
        if day < cutoff:
            continue
        action = r.get("action")
        if action == "bet":
            n_bet += 1
            stake += float(r.get("kelly") or 0)
        elif action == "shadow":
            n_shadow += 1
        if action in ("bet", "shadow") and r.get("clv") is not None:
            vals.append(float(r["clv"]))
    return {
        "n_bet": int(n_bet),
        "n_shadow": int(n_shadow),
        "stake_total": round(stake, 4),
        "avg_clv": round(sum(vals) / len(vals), 4) if vals else None,
        "n_clv": len(vals),
        "days": days,
    }


def format_daily_digest(*, days: int = 1) -> str:
    """Testo digest Telegram: volume, stake, CB, CLV."""
    from modules.advisor.online_learn import effective_min_edge
    from modules.advisor.risk_controls import circuit_breaker_status

    try:
        from modules.advisor.health_report import build_health_report

        health = build_health_report(days=max(7, days), refresh_metrics=False)
    except Exception:
        health = {}

    cb = circuit_breaker_status()
    clv = _avg_clv_recent(days=max(7, days))
    summary = (health or {}).get("summary") or {}
    when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

    stake_today = 0.0
    n_upcoming_bet = 0
    try:
        up_path = ROOT / "data" / "processed" / "upcoming_predictions.json"
        if up_path.is_file():
            ups = json.loads(up_path.read_text(encoding="utf-8"))
            today = datetime.now(TZ).date().isoformat()
            for p in ups:
                if p.get("action") != "bet":
                    continue
                if str(p.get("date") or "")[:10] != today:
                    continue
                rec = p.get("recommended") or {}
                stake_today += float(rec.get("kelly") or 0)
                n_upcoming_bet += 1
    except Exception:
        pass

    roi_v = summary.get("roi_kelly")
    if roi_v is None:
        roi_v = summary.get("roi_flat")
    roi_line = (
        f"Performance tip TG: ROI {100.0 * float(roi_v):+.1f}% sul puntato "
        f"| P&L {100.0 * float(summary.get('roi_pnl') or 0):+.1f}% bankroll "
        f"(n={summary.get('roi_n_settled', 0)})"
        if roi_v is not None
        else f"Performance tip TG: n/d (n={summary.get('roi_n_settled', 0)})"
    )

    lines = [
        f"{BRAND} — digest",
        when + " Roma",
        "",
        f"Oggi: {n_upcoming_bet} tip da giocare | stake totale ≈ {stake_today:.2%} BR",
        f"7g: {summary.get('bets_last_7d', clv['n_bet'])} bet inviato / "
        f"{summary.get('shadow_last_7d', clv['n_shadow'])} shadow",
        roi_line,
        f"CB={'ON' if cb.get('active') else 'off'} | "
        f"edge min {float(effective_min_edge()):.1%} | "
        f"BCR {summary.get('bcr_quality_pct')}% (n={summary.get('bcr_quality_n')})",
    ]
    return "\n".join(lines)


def dispatch_daily_digest(*, dry_run: bool = False, force: bool = False) -> dict:
    """Invia (max 1/giorno) il digest salute su Telegram."""
    sent_ids = _load_sent()
    day_key = f"digest|{datetime.now(TZ).date().isoformat()}"
    if not force and day_key in sent_ids:
        info = {"skipped": True, "reason": "already_sent_today", "key": day_key}
        print(f"telegram digest: skip ({day_key})")
        return info

    text = format_daily_digest()
    info: dict = {
        "skipped": False,
        "dry_run": dry_run,
        "key": day_key,
        "n_sent": 0,
        "chars": len(text),
        "status": telegram_status(),
    }
    if dry_run:
        print(text)
        print("---")
        return info

    if not load_credentials():
        print("telegram digest skip: credenziali assenti")
        info["error"] = "no_credentials"
        return info

    if send_message(text):
        sent_ids[day_key] = _now().isoformat()
        _save_sent(sent_ids)
        info["n_sent"] = 1
        print("telegram digest: inviato")
    else:
        info["error"] = "send_failed"
        print("telegram digest: invio fallito")
    return info

