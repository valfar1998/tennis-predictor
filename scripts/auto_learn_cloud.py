"""Cloud: sync risultati → settle → online learn → audit BCR (+ Telegram opzionale)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def notify_learn_summary(*, settle: dict, learn: dict | None, audit: dict) -> bool:
    from modules.advisor.validation_freeze import is_frozen
    from modules.notify.alerts import BRAND, brand_header
    from modules.notify.telegram import load_credentials, send_message

    if not load_credentials():
        return False

    ol = (learn or {}).get("online_learn") or audit.get("online_learn") or {}
    bcr = audit.get("bcr_pinnacle") or {}
    bcr_txt = (
        f"{bcr.get('bcr_pct')}% ({bcr.get('beats')}/{bcr.get('n')})"
        if bcr.get("n")
        else "campione insufficiente"
    )

    text = (
        f"{brand_header()}\n\n"
        f"📊 {BRAND} — Auto Learn\n\n"
        f"Settle: {settle.get('settled', 0)} pick chiuse\n"
        f"Pick settle totali: {learn.get('n_settled') if learn else '—'}\n"
        f"Hit rate: {(learn.get('hit_rate') if learn else None) or '—'}\n"
        f"BCR Pinnacle: {bcr_txt}\n"
    )
    if is_frozen():
        text += "🔒 FREEZE validazione: pesi invariati (solo metriche)\n"
    else:
        text += (
            f"MIN_EDGE appreso: {ol.get('min_edge_suggested', '—')}\n"
            f"Soglia alert: {ol.get('alert_min_suggested', '—')}\n"
            f"Dropping boost: {ol.get('dropping_boost', 0)}\n"
            f"Moneyway boost: {ol.get('moneyway_boost', 0)}"
        )
    return bool(send_message(text))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Settle + online learn (cloud)")
    parser.add_argument("--notify", action="store_true", help="Riepilogo Telegram post-learn")
    args = parser.parse_args()

    from modules.data_update.history import settle_pending
    from modules.advisor.live_metrics import format_bcr_status, run_live_audit
    from modules.advisor.validation_freeze import format_freeze_banner
    from modules.ops_progress import OpProgress, log_done

    print(format_freeze_banner())
    prog = OpProgress(4 if args.notify else 3, label="auto-learn")
    prog.next("Settle + online learn...")
    settle_out = settle_pending(learn=True)
    learn_out = settle_out.get("online_learn")
    prog.next("Report online learn...")
    if isinstance(learn_out, dict):
        print("Online learn:", json.dumps(learn_out, indent=2, ensure_ascii=False))
    elif settle_out.get("online_learn_error"):
        print("Online learn skip:", settle_out["online_learn_error"])

    prog.next("Audit BCR...")
    audit = run_live_audit(refresh_slippage=True)
    print(format_bcr_status(audit.get("bcr_pinnacle", {})))
    print(json.dumps({"settle": settle_out, "audit_phase": audit.get("phase")}, indent=2, default=str))

    if args.notify:
        prog.next("Telegram riepilogo...")
        sent = notify_learn_summary(
            settle=settle_out,
            learn=learn_out if isinstance(learn_out, dict) else None,
            audit=audit,
        )
        print("telegram learn summary:", "sent" if sent else "skip")
    log_done("auto-learn completato")


if __name__ == "__main__":
    main()
