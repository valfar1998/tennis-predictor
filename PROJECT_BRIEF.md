# PROJECT BRIEF — Tennis Predictor

## Visione

Sistema predittivo tennis da zero, adattando il framework calcistico esistente. Paradigma: interazioni individuali ad alta frequenza (punti/game) vs gol nel calcio.

## Stack

- Python 3.10+, pandas, XGBoost, Streamlit, DuckDB-ready
- Solo fonti gratuite: Sackmann, tennis-data.co.uk, Tennis Abstract, UTS, Open-Meteo, MCP

## Layer predittivo (3 livelli)

1. **Elo multisuperficie** — K adattivo, decay inattività, blend global/surface (w≈0.68)
2. **Markov** — Barnett & Clarke: P(serve) → game → tiebreak ABBA → set → BO3/BO5
3. **ML** — XGBoost con fatica, H2H, ranking Δ, form superficie, livello torneo

## Betting

- De-vig: Shin (default) o Power
- EV = P_model × quota - 1
- Kelly: γ=0.20, cap 1.8% bankroll
- Regole ritiro: matrice per bookmaker (1-ball, 1-set, full void)

## Metriche chiave

- **CLV** vs Pinnacle close — unico indicatore affidabile di edge
- ROI, hit rate, realization factor, max drawdown
- Breakdown per superficie e livello torneo

## Roadmap MVP

| Fase | Settimana | Deliverable |
|------|-----------|-------------|
| 1 | 1 | Data ingestion + Elo engine |
| 2 | 2 | Markov + Shin de-vig |
| 3 | 3 | ML + backtest 2018-2025 |
| 4 | 4 | Live automation + Telegram |

## Non in scope (richiedono account a pagamento)

- Betfair Exchange API
- Pinnacle API nativa
- TheOddsAPI (tier free limitato)
- Apify actors
