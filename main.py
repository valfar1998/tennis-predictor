"""Orchestratore CLI: dati → feature → training → prediction → backtest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def cmd_sync(args: argparse.Namespace) -> None:
    from modules.ops_progress import OpProgress, log_done
    from modules.data_update.sackmann import sync_sackmann_atp, sync_sackmann_wta
    from modules.data_update.tml import sync_tml
    from modules.data_update.player_registry import registry_stats, sync_sackmann_players, sync_tml_players
    from modules.data_update.tennis_data_odds import download_tennis_data_odds
    from modules.data_update.charting import sync_charting_data
    from modules.data_update.tennis_abstract import fetch_all_tennis_abstract_elo
    from modules.data_update.uts import fetch_uts_cpi
    from modules.data_update.courtspeed import fetch_courtspeed_cpi
    from modules.data_update.tennisratio import fetch_tennisratio_rankings
    from modules.data_update.infotennis import sync_infotennis_data
    from modules.data_update.tour_feeds import sync_tour_feeds
    from modules.data_update.wikidata import enrich_players_from_wikidata
    from modules.data_update.seeder_bridge import export_seeder_data
    import pandas as pd

    base_steps = 6 + (8 if args.extra else 0)
    prog = OpProgress(base_steps, label="sync")

    prog.next("TML Database...")
    print(sync_tml(clone=args.copy, pull=True))
    try:
        prog.next("Player registry...")
        print("Player registry ATP:", sync_sackmann_players(tour="ATP"))
        print("Player registry WTA:", sync_sackmann_players(tour="WTA"))
        print("Player registry TML:", sync_tml_players())
        print("Registry stats:", registry_stats())
    except Exception as exc:
        print(f"Player registry skip: {exc}")
    prog.next("Sackmann ATP...")
    print("Sackmann ATP:", sync_sackmann_atp(copy=args.copy))
    try:
        prog.next("Sackmann WTA...")
        print("Sackmann WTA:", sync_sackmann_wta(copy=args.copy))
    except FileNotFoundError as exc:
        from modules.data_update.sackmann import ensure_sackmann_wta
        ensured = ensure_sackmann_wta(clone=True)
        print(f"Sackmann WTA auto: {ensured}")
        if not ensured.get("ok"):
            print(f"Sackmann WTA skip: {exc}")
    prog.next("Tennis-data odds...")
    print("Tennis-data odds:", download_tennis_data_odds(force=args.force))
    try:
        prog.next("Tennis livescore...")
        from modules.data_update.tennis_livescore import fetch_tennis_livescore

        print("Tennis livescore:", fetch_tennis_livescore(force=args.force))
    except Exception as exc:
        print(f"Tennis livescore skip: {exc}")
    if args.extra:
        prog.next("Charting MCP...")
        print("Charting MCP:", sync_charting_data(force=args.force, copy=True))
        prog.next("Tennis Abstract Elo...")
        print("Tennis Abstract Elo:", fetch_all_tennis_abstract_elo(force=args.force))
        prog.next("CourtSpeed CPI...")
        print("CourtSpeed CPI:", fetch_courtspeed_cpi(force=args.force))
        prog.next("UTS CPI (fallback)...")
        print("UTS CPI (fallback):", fetch_uts_cpi(force=args.force))
        prog.next("TennisRatio skills...")
        print("TennisRatio skills:", fetch_tennisratio_rankings(force=args.force))
        prog.next("InfoTennis...")
        print("InfoTennis:", sync_infotennis_data())
        prog.next("Tour feeds...")
        print("Tour feeds:", sync_tour_feeds(force=args.force))
        seeder = export_seeder_data()
        print("Seeder bridge:", seeder)
        players_path = ROOT / "data" / "raw" / "atp" / "atp_players.csv"
        if players_path.exists():
            players = pd.read_csv(players_path, nrows=500)
            print("Wikidata:", enrich_players_from_wikidata(players, force=args.force))
    log_done("sync completato")


def cmd_build(args: argparse.Namespace) -> None:
    from modules.dataset_loader import DatasetLoader
    loader = DatasetLoader(min_year=args.min_year)
    df = loader.build()
    print(f"matches.csv: {len(df)} righe -> data/processed/matches.csv")


def cmd_features(args: argparse.Namespace) -> None:
    from modules.feature_engineering.feature_store import FEATURES_PARQUET, build_feature_store
    df = build_feature_store(force=args.force)
    print(f"features: {len(df)} righe -> {FEATURES_PARQUET}")


def cmd_retrain(args: argparse.Namespace) -> None:
    from scripts.run_retrain_pipeline import run_full_ml_pipeline
    result = run_full_ml_pipeline(min_year=args.min_year, force_features=args.force)
    print(json.dumps(result, indent=2, default=str))


def cmd_train(args: argparse.Namespace) -> None:
    from modules.model_training import ModelTrainer
    trainer = ModelTrainer()
    metrics = trainer.train()
    print("Training completato:", json.dumps(metrics, indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    from modules.calibration import run_backtest
    result = run_backtest()
    print(json.dumps(result, indent=2, default=str))


def cmd_scrape_oddsportal(args: argparse.Namespace) -> None:
    from modules.data_update.oddsportal_scraper import scrape_oddsportal_close

    result = scrape_oddsportal_close(
        max_matches=args.max_matches,
        enrich_h2h=not args.no_h2h,
        headless=not args.headed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_learn(args: argparse.Namespace) -> None:
    from modules.advisor.live_metrics import format_bcr_status, run_live_audit
    from modules.advisor.validation_freeze import format_freeze_banner
    from modules.data_update.history import history_summary, settle_pending
    from modules.ops_progress import OpProgress, log_done

    print(format_freeze_banner())
    prog = OpProgress(3 if args.no_learn else 4, label="learn")
    prog.next("Settle pick pendenti...")
    out = settle_pending(learn=not args.no_learn)
    print(json.dumps(out, indent=2, default=str))
    prog.next("Riepilogo storico...")
    print("Summary:", json.dumps(history_summary(), indent=2))
    prog.next("Audit BCR + slippage...")
    audit = run_live_audit(refresh_slippage=True)
    print(format_bcr_status(audit.get("bcr_betfair", {})))
    if audit.get("slippage", {}).get("recommendation"):
        print("Slippage:", audit["slippage"]["recommendation"])
    log_done("learn completato")


def cmd_metrics(args: argparse.Namespace) -> None:
    from modules.advisor.live_metrics import format_bcr_status, run_live_audit
    from modules.advisor.validation_freeze import format_freeze_banner
    from modules.data_update.history import refresh_clv_close

    print(format_freeze_banner())
    bcr_days = getattr(args, "days", None)
    if getattr(args, "refresh_clv", False):
        clv = refresh_clv_close(
            days=bcr_days or 14,
            include_settled=bool(bcr_days),
        )
        print("CLV refresh:", json.dumps(clv, ensure_ascii=False))
    audit = run_live_audit(refresh_slippage=not args.no_slippage, bcr_days=bcr_days)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(format_bcr_status(audit.get("bcr_betfair", {})))
    slip = audit.get("slippage") or {}
    if slip.get("recommendation"):
        print("Slippage:", slip["recommendation"])
    print(f"Report: data/processed/live_metrics.json")


def cmd_predict(args: argparse.Namespace) -> None:
    from modules.advisor.live_metrics import format_bcr_status, run_live_audit
    from modules.advisor.risk_controls import circuit_breaker_status
    from modules.advisor.validation_freeze import format_freeze_banner
    from modules.data_update.upcoming import build_upcoming
    from modules.notify.alerts import dispatch_alerts

    print(format_freeze_banner())
    cb = circuit_breaker_status()
    if cb["active"]:
        print(
            f"Circuit breaker ATTIVO — MIN_EDGE {cb['base_min_edge']:.1%} → {cb['min_edge']:.1%} "
            f"(DD {cb['current_drawdown']:.1%}, streak {cb['streak_loss_units']:.1f} unit)"
        )

    try:
        from modules.advisor.slippage_audit import refresh_slippage_snapshots

        refresh_slippage_snapshots()
    except Exception:
        pass

    preds = build_upcoming()
    print(f"Predizioni: {len(preds)}")
    bets = [p for p in preds if p.get("action") == "bet"]
    print(f"Value bet: {len(bets)}")
    if args.notify and bets:
        result = dispatch_alerts(preds)
        print(f"Alert Telegram inviati: {result.get('n_sent', 0)}")
    out = ROOT / "data" / "processed" / "upcoming_predictions.json"
    print(f"Salvato in {out}")

    if args.metrics:
        audit = run_live_audit(refresh_slippage=False)
        print(format_bcr_status(audit.get("bcr_betfair", {})))


def cmd_full(args: argparse.Namespace) -> None:
    cmd_sync(args)
    cmd_build(args)
    cmd_features(args)
    cmd_train(args)
    cmd_backtest(args)
    cmd_predict(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tennis Predictor — value betting system")
    sub = parser.add_subparsers(dest="command")

    p_sync = sub.add_parser("sync", help="Sincronizza dati (Sackmann, odds, scrapers)")
    p_sync.add_argument("--copy", action="store_true", help="Copia CSV Sackmann in data/raw/atp")
    p_sync.add_argument("--force", action="store_true")
    p_sync.add_argument("--extra", action="store_true", help="Scarica anche MCP, TA Elo, UTS CPI")
    p_sync.set_defaults(func=cmd_sync)

    p_build = sub.add_parser("build", help="Costruisce matches.csv")
    p_build.add_argument("--min-year", type=int, default=2000)
    p_build.set_defaults(func=cmd_build)

    p_feat = sub.add_parser("features", help="Calcola feature store (Parquet + CSV)")
    p_feat.add_argument("--force", action="store_true", help="Ricalcola anche se cache valida")
    p_feat.set_defaults(func=cmd_features)

    p_retrain = sub.add_parser("retrain", help="Pipeline ML completa (TML + feature store + stacker)")
    p_retrain.add_argument("--min-year", type=int, default=2010)
    p_retrain.add_argument("--force", action="store_true", help="Ricalcola feature store")
    p_retrain.set_defaults(func=cmd_retrain)

    sub.add_parser("train", help="Training XGBoost").set_defaults(func=cmd_train)
    sub.add_parser("backtest", help="Backtest su OOF").set_defaults(func=cmd_backtest)

    p_pred = sub.add_parser("predict", help="Genera predizioni upcoming")
    p_pred.add_argument("--notify", action="store_true", help="Invia alert Telegram")
    p_pred.add_argument("--metrics", action="store_true", help="Stampa BCR/slippage dopo predict")
    p_pred.set_defaults(func=cmd_predict)

    p_metrics = sub.add_parser("metrics", help="Audit fase live: BCR Betfair + slippage Telegram")
    p_metrics.add_argument("--no-slippage", action="store_true", help="Non aggiornare snapshot T+3min")
    p_metrics.add_argument("--days", type=int, default=None, help="Finestra BCR (es. 3 = ultimi 3 giorni)")
    p_metrics.add_argument(
        "--refresh-clv",
        action="store_true",
        help="Ricalcola CLV/BCR da quote chiusura (con --days include pick già settle)",
    )
    p_metrics.set_defaults(func=cmd_metrics)

    p_learn = sub.add_parser("learn", help="Chiude pick pendenti e aggiorna calibration.json")
    p_learn.add_argument("--no-learn", action="store_true", help="Solo settle, senza online learn")
    p_learn.set_defaults(func=cmd_learn)

    p_op = sub.add_parser("scrape-oddsportal", help="Scrape quote chiusura OddsPortal (Playwright)")
    p_op.add_argument("--max-matches", type=int, default=80)
    p_op.add_argument("--no-h2h", action="store_true")
    p_op.add_argument("--headed", action="store_true")
    p_op.set_defaults(func=cmd_scrape_oddsportal)

    p_full = sub.add_parser("full", help="Pipeline completa")
    p_full.add_argument("--copy", action="store_true")
    p_full.add_argument("--force", action="store_true")
    p_full.add_argument("--extra", action="store_true")
    p_full.add_argument("--min-year", type=int, default=2000)
    p_full.add_argument("--notify", action="store_true")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
