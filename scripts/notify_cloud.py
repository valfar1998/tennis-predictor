"""Cloud: Betfair + predizioni + alert Telegram (job 30 min e giornaliero)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.ops_progress import OpProgress, log_done


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from modules.data_update.upcoming import build_upcoming
    from modules.notify.alerts import dispatch_alerts
    from modules.notify.telegram import send_message, telegram_status

    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    prog = OpProgress(5, label="cloud")
    print(telegram_status(), flush=True)

    info: dict = {"cloud": True}
    try:
        from modules.data_update.betfair import fetch_betfair_odds

        prog.next("Betfair odds...")
        bf = fetch_betfair_odds(force=True, days=7)
        info["betfair_events"] = bf.get("n_events", 0)
        info["betfair_from_cache"] = bf.get("from_cache", False)
        info["betfair_ok"] = bool(bf.get("ok"))
        if not bf.get("ok") and bf.get("error"):
            info["betfair_error"] = bf["error"]
            info["betfair_soft_fail"] = True
            print(f"  betfair_soft_fail: {bf['error']}", flush=True)
    except Exception as exc:
        info["betfair_error"] = str(exc)
        info["betfair_soft_fail"] = True
        info["betfair_ok"] = False
        print(f"  betfair_soft_fail: {exc}", flush=True)

    try:
        from modules.data_update.market_signals import sync_market_signals

        prog.next("Market signals...")
        sig = sync_market_signals(force=True)
        info["market_signals"] = sig
    except Exception as exc:
        info["market_signals_error"] = str(exc)

    try:
        from modules.data_update.history import settle_pending

        prog.next("Settle pending...")
        info["settle"] = settle_pending(learn=True)
    except Exception as exc:
        info["settle_error"] = str(exc)

    prog.next("Build upcoming predictions...")
    preds = build_upcoming(use_betfair=True)
    print(f"  predictions={len(preds)}", flush=True)
    prog.next("Telegram alerts...")
    alerts = dispatch_alerts(preds)
    info.update({
        "n_predictions": len(preds),
        "n_bets": alerts.get("n_bets", 0),
        "n_new_bets": alerts.get("n_new_bets", 0),
        "n_alerted": alerts.get("n_sent", 0),
    })
    log_done("notify_cloud completato")
    print(json.dumps(info, indent=2, default=str), flush=True)


def notify_run_status(*, status: str, run_id: str, repo: str) -> None:
    """Notifica esito job giornaliero su Telegram."""
    from modules.notify.alerts import BRAND, brand_header
    from modules.notify.telegram import send_message

    icon = "✅" if status.lower() == "success" else "❌"
    run_url = f"https://github.com/{repo}/actions/runs/{run_id}" if repo and run_id else ""
    text = (
        f"{brand_header()}\n\n"
        f"{icon} {BRAND} — Aggiorna dati e modello\n\n"
        f"Esito: {status}\n"
        f"Run: {run_url}"
    )
    send_message(text)


if __name__ == "__main__":
    main()
