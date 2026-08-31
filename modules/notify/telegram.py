"""Bot Telegram (stesso .env di telegram-offerte-sconto / football-predictor)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIBLING_ENV_DIRS = (
    "telegram-offerte-sconto",
    "offerte_notifications",
    "offerte-notifications",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _env_candidates() -> list[Path]:
    paths = [ROOT / ".env"]
    parent = ROOT.parent
    extra = os.getenv("OFFERTE_NOTIFICATIONS_DIR", "").strip()
    if extra:
        paths.append(Path(extra) / ".env")
    for name in SIBLING_ENV_DIRS:
        paths.append(parent / name / ".env")
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _clean_secret(value: str) -> str:
    return "".join(str(value or "").split())


def load_credentials() -> dict[str, str] | None:
    token = _clean_secret(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat = _clean_secret(os.getenv("TELEGRAM_CHAT_ID", ""))
    if token and chat:
        return {"token": token, "chat_id": chat, "source": "env"}

    for path in _env_candidates():
        data = _parse_env_file(path)
        token = token or _clean_secret(data.get("TELEGRAM_BOT_TOKEN", ""))
        chat = chat or _clean_secret(data.get("TELEGRAM_CHAT_ID", ""))
        if token and chat:
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", token)
            os.environ.setdefault("TELEGRAM_CHAT_ID", chat)
            return {"token": token, "chat_id": chat, "source": str(path)}
    return None


def telegram_status() -> str:
    creds = load_credentials()
    if not creds:
        return "Telegram: manca TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID."
    src = creds["source"]
    where = "variabili d'ambiente" if src == "env" else Path(src).parent.name
    return f"Telegram: pronto ({where}). Value bet con EV positivo."


def send_message(text: str, *, delay: float = 0.5, parse_mode: str | None = None) -> bool:
    creds = load_credentials()
    if not creds:
        print("telegram skip: credenziali assenti")
        return False
    url = f"https://api.telegram.org/bot{creds['token']}/sendMessage"
    payload: dict[str, str] = {
        "chat_id": creds["chat_id"],
        "text": text[:4000],
        "disable_web_page_preview": "true",
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    body = urllib.parse.urlencode(payload).encode()
    if delay > 0:
        time.sleep(delay)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data.get("ok"):
            print(f"telegram errore: {data.get('description') or data}")
            return False
        return True
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"telegram errore: {exc}")
        return False


def send_telegram(message: str) -> dict:
    """Compatibilità con chiamate legacy."""
    ok = send_message(message)
    return {"ok": ok}
