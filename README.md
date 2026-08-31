# Tennis Predictor — Sistema Predittivo Value Betting

Sistema end-to-end per il tennis ATP: Elo multisuperficie, modello Markov punto→match, layer ML, de-vig Shin/Power, Kelly frazionario e backtesting con CLV.

Adattato dall'architettura di [football-predictor](../football-predictor).

## Architettura

```
Data (Sackmann + tennis-data + scrapers gratuiti)
  → Entity Resolution → Feature Engine (Elo, H2H, fatica)
  → Markov + ML → Devigging Shin → EV/Kelly → Backtest/Alert
```

### Moduli

| Modulo | Descrizione |
|--------|-------------|
| **Data Pipeline** | Jeff Sackmann ATP, tennis-data.co.uk, MCP charting, Tennis Abstract Elo, UTS CPI |
| **Entity Resolution** | Fuzzy matching nomi giocatori (rapidfuzz) |
| **Elo Engine** | Multisuperficie, K-factor adattivo, inactivity decay |
| **Markov Chain** | P(serve) → game → tiebreak → set → match (BO3/BO5) |
| **ML Layer** | XGBoost binario con feature contestuali |
| **Betting Engine** | Shin/Power de-vig, EV, Kelly 1/5, regole ritiro |
| **Backtest** | Walk-forward OOF, ROI, CLV, per superficie |

### Fonti dati gratuite

| Fonte | Modulo | Uso |
|-------|--------|-----|
| **Jeff Sackmann** | `sackmann.py` | Match storici ATP 1968-2026 |
| **tennis-data.co.uk** | `tennis_data_odds.py` | Quote storiche Pinnacle/B365 |
| **CourtSpeed.com** | `courtspeed.py` | CPI campo (primario) |
| **Tennis Abstract** | `tennis_abstract.py` | Elo per superficie |
| **UTS** | `uts.py` | CPI fallback |
| **Match Charting Project** | `charting.py` | Stats servizio colpo-per-colpo |
| **TennisRatio** | `tennisratio.py` | Skill ratings ATP/WTA |
| **Wikidata SPARQL** | `wikidata.py` | Altezza, mano, paese giocatori |
| **InfoTennis** | `infotennis.py` | Keystats ATP 2021+ (path locale) |
| **Seeder** | `seeder_bridge.py` | Odds TennisExplorer (opzionale) |
| **Open-Meteo** | `weather.py` | Meteo pre-match |
| **Altitudine** | `altitude.py` | Correzione tornei ad alta quota |

**Non integrati** (Cloudflare/account): atptour.com diretto, wtatennis.com diretto, Pinnacle API.
**Quote live:** Betfair Exchange (`betfair.py`) per value bet vs modello.
Per ATP live usa **infotennis** o **seeder** come bridge.

## Setup

```bash
cd tennis-predictor
pip install -r requirements.txt
cp .env.example .env
# Scarica i repo dati in lib/ (vedi PROJECT_BRIEF) oppure lascia i default lib/
```

## Utilizzo

```bash
# Pipeline completa
python main.py full --copy --extra

# Singoli step
python main.py sync --copy          # Dati Sackmann + odds
python main.py build                # matches.csv
python main.py features             # features.csv
python main.py train                # XGBoost + OOF
python main.py backtest             # Simulazione value bet
python main.py predict --notify     # Predizioni + Telegram

# Dashboard
streamlit run app.py
```

## Struttura

```
tennis-predictor/
├── main.py                 # CLI orchestrator
├── app.py                  # Dashboard Streamlit
├── lib/                    # Repo dati esterni (Sackmann, MCP, infotennis, …)
├── modules/
│   ├── lib_paths.py        # Default path sotto lib/
│   ├── data_update/        # Ingestion, scrapers, weather, altitude
│   ├── dataset_loader/     # Merge match + odds
│   ├── feature_engineering/# Elo, H2H, fatica, form
│   ├── markov/             # Catena punto→match
│   ├── model_training/     # XGBoost
│   ├── predictor/          # Inference blend
│   ├── advisor/            # EV, Kelly, de-vig
│   ├── calibration/        # Backtest, config
│   └── notify/             # Telegram
└── data/
    ├── raw/                # CSV Sackmann, odds, scrapers
    ├── processed/          # matches.csv, features.csv
    └── models/             # best_model.joblib, calibration.json
```

## Licenza dati

I dati Jeff Sackmann sono CC BY-NC-SA 4.0 (uso non commerciale).
