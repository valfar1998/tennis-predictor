"""Alert Telegram per value bet (opzionale)."""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def send_telegram(message: str) -> dict:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID mancanti"}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
    req = Request(url, data=data, method="POST")
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def alert_value_bets(predictions: list[dict]) -> int:
    sent = 0
    for pred in predictions:
        rec = pred.get("recommended")
        if not rec:
            continue
        msg = (
            f"🎾 <b>Value Bet Tennis</b>\n"
            f"{pred.get('player_a')} vs {pred.get('player_b')}\n"
            f"Pick: <b>{rec.get('player')}</b> @ {rec.get('odds')}\n"
            f"P={rec.get('probability', 0):.1%} | EV={rec.get('ev', 0):+.1%} | Kelly={rec.get('kelly', 0):.2%}\n"
            f"Superficie: {pred.get('surface')} | {pred.get('tourney', '')}"
        )
        result = send_telegram(msg)
        if result.get("ok"):
            sent += 1
    return sent
