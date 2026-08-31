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

- Pinnacle API nativa
- TheOddsAPI (tier free limitato)
- Apify actors

*(Betfair Exchange è integrato opzionalmente via `.env` per quote live.)*

### Setup Elo WTA

1. **Sackmann WTA** (opzionale, migliora Elo storico): clona [JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta) in `data/raw/wta/` oppure imposta `SACKMANN_WTA_PATH` nel `.env`
2. **Tennis Abstract WTA** (automatico): scaricato a ogni `predict` → `data/raw/tennis_abstract_elo_wta.json`
3. **Tour detection**: tornei con "Women's" / "WTA" usano engine WTA separato

Senza CSV Sackmann WTA, il sistema usa **Tennis Abstract + nomi charting** (fallback TA-only).

---

## Pipeline di analisi (live)

Comando principale: `python main.py predict` (o pulsante **Aggiorna calendario** in Streamlit, porta 8502).

### 1. Ingestion dati e segnali mercato

| Step | Modulo | Output |
|------|--------|--------|
| Match storici Sackmann ATP | `sackmann.py` | Elo engine ATP (`data/raw/atp/`) |
| Match storici Sackmann WTA | `sackmann.py` | Elo engine WTA (`data/raw/wta/`) |
| Elo Tennis Abstract | `tennis_abstract.py` | ATP + WTA cache JSON |
| Quote live | `betfair.py` | `data/raw/betfair_odds.json` |
| Moneyway volume | `market_signals.py` → [Arbworld 1x2](https://arbworld.net/moneyway/tennis/1x2) | `arbworld_moneyway.json` |
| Dropping odds | `market_signals.py` → [OddsSafari sport 30](https://www.oddssafari.com/dropping-odds/sports/30) | `oddssafari_dropping.json` |
| OddsPortal | *non integrato* — vedi sezione [Fonti mercato esterne](#fonti-mercato-esterne) | — |
| Entity resolution | `entity_resolution.py` | Nomi Betfair → ATP + WTA (charting) |

Cache segnali mercato: **30 min**. Betfair: **1 h** (o cache se API non disponibile).

### 2. Predizione per ogni match

Per ogni evento Betfair (fallback: match recenti Sackmann con quote storiche):

1. **Tour detection** — `Women's US Open` → WTA, `Men's US Open` → ATP
2. **Risoluzione giocatori** — alias, fuzzy match, formato abbreviato (`H Dart` → `Harriet Dart`)
3. **Elo** — engine separato ATP/WTA da Sackmann + blend Tennis Abstract per tour
3. **Markov** — P(hold serve) → probabilità vittoria match (BO3/BO5)
4. **ML** — XGBoost (se modello trainato e feature disponibili)
5. **Blend** — `P(A)` = 40% Markov + 25% Elo + 35% ML (ML assente → solo Markov + Elo)

Output: `p_win_a`, `p_markov`, `p_elo`, `p_ml`, `tour` (ATP/WTA), flag `model_low_confidence` se giocatori non identificati.

### 3. Value bet (consiglio scommessa)

Modulo `advise.py` + `value.py`:

1. **De-vig Shin** sulle quote → probabilità di mercato fair (`mkt_prob`)
2. **EV** per entrambi i lati: `EV = P_model × quota − 1` (es. 0.15 = **+15%**)
3. **Edge** in punti percentuali: `edge = P_model − mkt_prob`
4. **Kelly frazionato** (γ=0.20, cap 1.8%)
5. Scelta del lato con **EV più alto**

**Filtri no-bet** (bloccano la raccomandazione):

| Filtro | Soglia |
|--------|--------|
| EV minimo | ≥ 2.5% (`MIN_EDGE`) |
| Probabilità minima | ≥ 38% (`MIN_PROB_PLAY`) |
| Modello incerto | giocatore/i non identificati nel database |
| Artefatto 50/50 | `P ≈ 50%` e `P_elo ≈ 50%` senza ML |

Se passa i filtri → `action: "bet"` + `recommended` (pick, quota, `ev`, `ev_pct`, Kelly).

Campi EV nel JSON:
- `ev` — decimale (0.152 = 15.2% di rendimento atteso per unità puntata)
- `ev_pct` — stesso valore in percentuale (+15.2)
- UI e Telegram mostrano sempre **EV %** (es. `+15.2%`)

### 4. Indice giocabilità (0–100)

Modulo `playability.py`. Calcolato **dopo** il value bet, arricchisce ogni predizione.

**Formula composita** (ogni componente normalizzata 0–1, poi × peso):

```
Giocabilità = 100 × (
    0.30 × value
  + 0.18 × model_agreement
  + 0.12 × kelly
  + 0.15 × market_quality
  + 0.13 × moneyway
  + 0.12 × dropping_odds
)
```

**Penalità finali:**
- Se `action ≠ "bet"` → score max **55**
- Se `EV < MIN_EDGE` → score max **45**

**Bande:**

| Score | Band | Significato |
|-------|------|-------------|
| 0–30 | No bet | Non giocabile |
| 30–60 | Lean | Interessante ma debole |
| 60–75 | Playable | Giocabile con cautela |
| 75–90 | Strong | Alert Telegram ✅ |
| 90–100 | Premium | Massima convinzione |

**Soglia alert Telegram:** `MIN_PLAY_ALERT = 75` (Strong+).

---

## Dettaglio componenti giocabilità

Ogni campo è normalizzato 0–1 prima di applicare il peso. Valori alti **alzano** la giocabilità; valori bassi la **abbassano**.

### 1. Value (peso 30%)

Misura quanto il pick supera la soglia di edge.

| Input | Effetto |
|-------|---------|
| **EV alto** (es. +15% → +1.0) | `(EV − 2.5%) / 12%`, clamp 0–1 |
| **Edge vs mercato** (es. +8 pp → +1.0) | `edge / 8%`, clamp 0–1 |
| Combinazione | 55% EV + 45% edge |
| EV sotto 2.5% | componente bassa; cap score a 45 |

### 2. Model agreement (peso 18%)

Accordo tra Markov, Elo e ML rispetto al blend finale.

| Input | Effetto |
|-------|---------|
| Modelli allineati (div < 15 pp) | → 1.0 |
| Divergenza media 15 pp | → 0.0 |
| ML assente | usa solo Markov + Elo |
| Modello ~50/50 senza ML | spesso blocca il bet prima (filtro incertezza) |

### 3. Kelly (peso 12%)

Stake consigliato rispetto al cap.

| Input | Effetto |
|-------|---------|
| Kelly = cap (1.8%) | → 1.0 |
| Kelly = 0 | → 0.0 |
| Formula | `kelly / KELLY_CAP` |

### 4. Market quality (peso 15%)

Qualità del mercato e del torneo.

| Sotto-componente | Peso interno | Cosa alza | Cosa abbassa |
|------------------|-------------|-----------|--------------|
| **Fonte quote** | 45% | Betfair 1.0, Pinnacle 0.95 | book generico 0.55 |
| **Overround** | 30% | mercato tight (ov ≈ 1.02) | overround largo (ov > 1.10) |
| **Livello torneo** | 25% | Grand Slam 1.0, Masters 0.85 | Challenger/ITF 0.45 |

Overround tight: `(1.08 − (overround − 1.0)) / 0.12`, clamp 0–1.

### 5. Moneyway (peso 13%) — Arbworld

Volume scommesso sul pick (steam Betfair aggregato).

| Input | Effetto |
|-------|---------|
| **Volume % sul pick** | `(vol% − 35%) / 55%` — sopra 35% aiuta, sotto abbassa |
| **Volume totale £** | liquidità: `min(1, £ / 5000)` |
| Combinazione | 70% volume pick + 30% liquidità |
| Match non trovato su Arbworld | neutro **0.5** |

Interpretazione: favorito con >70% volume = segnale sharp; underdog con poco volume ma EV alto non viene penalizzato eccessivamente.

### 6. Dropping odds (peso 12%) — OddsSafari

Movimento quote verso il nostro pick.

| Scenario | Score |
|----------|-------|
| Drop ≥10% **allineato** al pick | 0.55 + drop/40 (max ~1.0) |
| Drop 5–10% allineato | 0.45 + drop/50 |
| Drop ≥10% **contro** il pick | 0.35 − drop/80 (penalizza) |
| Nessun segnale / non trovato | neutro **0.5** |

Allineamento: pick lato A + dropping su `"1"`, oppure lato B + dropping su `"2"`.

---

## Output e notifiche

| Output | Path / canale |
|--------|---------------|
| Predizioni complete | `data/processed/upcoming_predictions.json` |
| Storico pick | `data/processed/our_history.sqlite` |
| Report apprendimento | `data/models/online_learn_report.json` |
| UI | Streamlit `app.py` — tab Calendario, ordinato per giocabilità |
| Telegram | `modules/notify/alerts.py` — solo `action=bet` **e** giocabilità **≥ 75**, dedup 21 gg |
| Cloud (GitHub Actions) | `scripts/notify_cloud.py` — sync Betfair + segnali + predict + alert |

Branding alert: **TENNIS_PREDICTOR**.

### Notifiche ogni 30 minuti (GitHub Actions)

Workflow: `.github/workflows/telegram-alerts.yml` — cron `*/30 * * * *`.

Ogni run esegue `scripts/notify_cloud.py`:

1. **Betfair** — quote live (cache 1 h)
2. **Segnali mercato** — Arbworld Moneyway + OddsSafari dropping (cache 30 min)
3. **Settle + learn** — chiude pick pendenti vs risultati Sackmann, aggiorna `calibration.json`
4. **Predict** — pipeline completa con giocabilità (inclusi moneyway 13% + dropping 12%)
5. **Telegram** — alert solo Strong+ (≥75), con dettaglio Moneyway/Drop

Cache persistenti tra run: `our_history.sqlite`, `telegram_alerts_sent.json`, segnali mercato, Betfair session.

Job giornaliero (`cloud-train.yml`, 07:00 IT): train XGBoost + stesso flusso notify + `cloud_learn.py`.

---

## Apprendimento da risultati passati

Moduli: `modules/data_update/history.py`, `modules/advisor/online_learn.py`.

### Archivio pre-match

Ogni predizione con `action=bet` viene salvata in SQLite con:

- pick, quota, EV, Kelly, probabilità modello
- **playability**, band, **tour**
- **moneyway_vol_pct**, **dropping_pct**, **dropping_aligned**

### Settle automatico

`settle_from_sackmann()` confronta pick pendenti con risultati Sackmann ATP/WTA (ultimi 14 gg):

- match per cognome + data (±2 giorni)
- calcola `hit` (1/0), `winner`, `score`

Comando locale: `python main.py learn` oppure `python scripts/cloud_learn.py`.

### Online learn

Con ≥12 pick chiuse, `learn_from_settled()` aggiorna `data/models/calibration.json`:

| Statistica | Uso |
|------------|-----|
| Hit rate per banda giocabilità | suggerisce soglia alert (75 vs 80) |
| ROI globale | suggerisce `min_edge` (2.5%–3.5%) |
| Hit rate drop allineato | `dropping_boost` fino a +8 pp su raw score |
| Hit rate con segnale Moneyway | `moneyway_boost` fino a +6 pp |

I boost vengono applicati in `playability.py` via `learned_playability_boost()`.

Report dettagliato: `data/models/online_learn_report.json`.

---

## Esecuzione rapida

```bash
# Analisi completa live
python main.py predict

# Analisi + invio Telegram (se credenziali in .env)
python main.py predict --notify

# Chiudi pick pendenti + apprendimento
python main.py learn

# UI locale
streamlit run app.py --server.port 8502
# oppure: apri_ui.bat
```

---

## Fonti mercato esterne

Tre siti utili per arricchire l'analisi. Stato attuale nel progetto:

| Fonte | URL | Utile? | Stato | Cosa aggiunge |
|-------|-----|--------|-------|---------------|
| **Arbworld Moneyway** | [arbworld.net/moneyway/tennis/1x2](https://arbworld.net/moneyway/tennis/1x2) | ✅ Sì | **Integrato** | Volume % scommesso su Betfair per lato, liquidità £ — componente **moneyway** (13%) della giocabilità |
| **OddsSafari Dropping** | [oddssafari.com/dropping-odds/sports/30](https://www.oddssafari.com/dropping-odds/sports/30) | ✅ Sì | **Integrato** | Quote in calo per match tennis — componente **dropping_odds** (12%). Drop allineato al pick alza lo score; drop contro lo abbassa |
| **OddsPortal** | [oddsportal.com/tennis](https://www.oddsportal.com/tennis/) | ⚠️ Utile ma difficile | **Non integrato** | Confronto multi-book, linee di apertura/chiusura, movimenti quote Pinnacle/B365 — ideale per **CLV** e overround cross-book |

### Perché OddsPortal non è ancora integrato

- Sito **JavaScript-heavy** (tabella renderizzata lato client): scraping HTML semplice insufficiente
- Servirebbe browser headless (Playwright) o API a pagamento
- **Betfair** copre già quote live per il value bet; **OddsSafari** copre i dropping

### Valore aggiunto se integrassimo OddsPortal

- Verificare se la nostra quota Betfair batte la **chiusura Pinnacle** (CLV)
- Penalizzare pick dove solo un book offre value (mercato non allineato)
- Conferma indipendente dei movimenti già visti su OddsSafari

**Priorità consigliata:** OddsPortal = nice-to-have; Arbworld + OddsSafari + Betfair coprono già il flusso MVP.
