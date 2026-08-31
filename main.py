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
    from modules.data_update.sackmann import sync_sackmann_atp, sync_sackmann_wta
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

    print("Sackmann ATP:", sync_sackmann_atp(copy=args.copy))
    try:
        print("Sackmann WTA:", sync_sackmann_wta(copy=args.copy))
    except FileNotFoundError as exc:
        from modules.data_update.sackmann import ensure_sackmann_wta
        ensured = ensure_sackmann_wta(clone=True)
        print(f"Sackmann WTA auto: {ensured}")
        if not ensured.get("ok"):
            print(f"Sackmann WTA skip: {exc}")
    print("Tennis-data odds:", download_tennis_data_odds(force=args.force))
    try:
        from modules.data_update.tennis_livescore import fetch_tennis_livescore

        print("Tennis livescore:", fetch_tennis_livescore(force=args.force))
    except Exception as exc:
        print(f"Tennis livescore skip: {exc}")
    if args.extra:
        print("Charting MCP:", sync_charting_data(force=args.force, copy=True))
        print("Tennis Abstract Elo:", fetch_all_tennis_abstract_elo(force=args.force))
        print("CourtSpeed CPI:", fetch_courtspeed_cpi(force=args.force))
        print("UTS CPI (fallback):", fetch_uts_cpi(force=args.force))
        print("TennisRatio skills:", fetch_tennisratio_rankings(force=args.force))
        print("InfoTennis:", sync_infotennis_data())
        print("Tour feeds:", sync_tour_feeds(force=args.force))
        seeder = export_seeder_data()
        print("Seeder bridge:", seeder)
        players_path = ROOT / "data" / "raw" / "atp" / "atp_players.csv"
        if players_path.exists():
            players = pd.read_csv(players_path, nrows=500)
            print("Wikidata:", enrich_players_from_wikidata(players, force=args.force))


def cmd_build(args: argparse.Namespace) -> None:
    from modules.dataset_loader import DatasetLoader
    loader = DatasetLoader(min_year=args.min_year)
    df = loader.build()
    print(f"matches.csv: {len(df)} righe -> data/processed/matches.csv")


def cmd_features(args: argparse.Namespace) -> None:
    from modules.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    df = fe.build()
    print(f"features.csv: {len(df)} righe")


def cmd_train(args: argparse.Namespace) -> None:
    from modules.model_training import ModelTrainer
    trainer = ModelTrainer()
    metrics = trainer.train()
    print("Training completato:", json.dumps(metrics, indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    from modules.calibration import run_backtest
    result = run_backtest()
    print(json.dumps(result, indent=2, default=str))


def cmd_learn(args: argparse.Namespace) -> None:
    from modules.data_update.history import history_summary, settle_pending

    out = settle_pending(learn=not args.no_learn)
    print(json.dumps(out, indent=2, default=str))
    print("Summary:", json.dumps(history_summary(), indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    from modules.data_update.upcoming import build_upcoming
    from modules.notify.alerts import dispatch_alerts

    preds = build_upcoming()
    print(f"Predizioni: {len(preds)}")
    bets = [p for p in preds if p.get("action") == "bet"]
    print(f"Value bet: {len(bets)}")
    if args.notify and bets:
        result = dispatch_alerts(preds)
        print(f"Alert Telegram inviati: {result.get('n_sent', 0)}")
    out = ROOT / "data" / "processed" / "upcoming_predictions.json"
    print(f"Salvato in {out}")


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

    sub.add_parser("features", help="Calcola features.csv").set_defaults(func=cmd_features)
    sub.add_parser("train", help="Training XGBoost").set_defaults(func=cmd_train)
    sub.add_parser("backtest", help="Backtest su OOF").set_defaults(func=cmd_backtest)

    p_pred = sub.add_parser("predict", help="Genera predizioni upcoming")
    p_pred.add_argument("--notify", action="store_true", help="Invia alert Telegram")
    p_pred.set_defaults(func=cmd_predict)

    p_learn = sub.add_parser("learn", help="Chiude pick pendenti e aggiorna calibration.json")
    p_learn.add_argument("--no-learn", action="store_true", help="Solo settle, senza online learn")
    p_learn.set_defaults(func=cmd_learn)

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
