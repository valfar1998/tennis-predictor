# PROJECT BRIEF — Tennis Predictor

## Visione

Sistema predittivo tennis da zero, adattando il framework calcistico esistente. Paradigma: interazioni individuali ad alta frequenza (punti/game) vs gol nel calcio.

## Stack

- Python 3.10+, pandas, XGBoost, Streamlit, DuckDB-ready
- Solo fonti gratuite: Sackmann, tennis-data.co.uk, Tennis Abstract, UTS, Open-Meteo, MCP

## Layer predittivo (3 livelli + meta-learner)

1. **Elo multisuperficie + CPI** — K adattivo, decay inattività, blend global/surface modulato dal **Court Speed Index** del torneo (`blended_with_cpi`: campo veloce → più peso rating superficie). **Serve-Elo / Return-Elo** disaggregati (`serve_return_elo.py`): P(serve) Markov da interazione Serve(A) vs Return(B), pesata da **CPI dinamico**
2. **Markov dinamico** — Barnett & Clarke con **P(serve) sotto pressione**: break point e tiebreak clutch da Match Charting Project (`modules/markov/pressure.py`). Input serve: **Serve-Elo A × Return-Elo B** (non più solo `elo_diff` monolitico). **Bayesian shrinkage**: se un giocatore ha **<5 match chartati** MCP, le statistiche sotto pressione regolarizzano verso la media circuito (`CIRCUIT_BP_SAVE ≈ 0.635`)
3. **ML (XGBoost)** — feature avanzate: fatica 7/14 gg, **viaggio Haversine**, jet lag, **hold vs break**, CPI, H2H, form. Training con **TimeSeriesSplit** (sort strict per `tourney_date`)
4. **Meta-learner (stacking)** — regressione logistica su **OOF temporale** (`TimeSeriesSplit`, mai K-Fold casuale) che sostituisce i pesi fissi 40/25/35 (`modules/model_training/stacker.py`). Calibrazione via **Brier OOF** (non in-sample)

Fonti dati esterne integrate / referenziate (cartelle in `lib/`):

| Repo locale (`lib/`) | Uso nel predictor |
|----------------------|-------------------|
| `tennis_MatchChartingProject-master` | BP save, tiebreak clutch, profilo servizio MCP |
| `tennis-sackmann-archive-main` | Storico ATP/WTA + settle |
| `tennis_atp-master` | Fallback ATP se manca sottocartella archive |
| `TML-Database-master` | Sorgente primaria ATP (merge con Sackmann gap-fill) |
| `infotennis-main` | Keystats ATP live (bridge scraper) |
| `seeder-main` | Odds TennisExplorer (SQLite opzionale) |
| `tennisgnn_predictions-main` | Benchmark Brier/ROI su clay 2025 |

## Betting

- De-vig: Shin (default) o Power
- **Calibrazione probabilità** (Isotonic, fallback Platt) su OOF stacker → `modules/calibration/prob_calibrator.py`; applicata in `predict.py` prima dello shrink di mercato
- **Prior Bayesiano di mercato**: `P_finale = w·P_modello + (1−w)·P_mercato` (`market_calibration.py`); `w` scende su ITF/bassa densità e se divergenza modello/mercato ≥12% (hard block >18%)
- EV = P_finale × quota − 1
- Kelly: γ=0.20, cap **dinamico per livello torneo** (`risk_controls.py` / `advise.py`)
- **Ranking pick**: Kelly-adjusted / Sharpe-like (`odds_sharpe`), **non** EV grezzo
- **Sanity EV**: hard discard se EV > **30%** (quote ≤3) o > **25%** (quote lunghe); fascia **>20%** → `action: review` (no alert auto)
- Regole ritiro: matrice per bookmaker (1-ball, 1-set, full void) + **sotto-modello P(ritiro)** (`retirement_risk.py`) che modula EV/Kelly

### Kelly cap per liquidità torneo

| Livello | Cap bankroll |
|---------|--------------|
| Grand Slam / Masters 1000 / Finals | **1.8%** |
| ATP/WTA Tour (250/500) | **1.2%** |
| Challenger | **1.0%** |
| ITF | **0.8%** |

### Controlli rischio esecuzione (`modules/advisor/risk_controls.py`)

| Controllo | Regola | Effetto |
|-----------|--------|---------|
| **Circuit breaker** | Drawdown corrente >15% **oppure** streak perdite ≥11 unità (1% ciascuna) | `MIN_EDGE` 2.5% → **4.5%** |
| **Esposizione giornaliera** | ≥6 bet stesso giorno + stesso torneo | Kelly scalato per cap totale **6%** bankroll |
| **EV sanity** | EV > 25–30% | `no_bet` (edge irrealistico) |
| **EV review** | 20% < EV ≤ cap | `action: review` |
| **Divergenza mkt** | \|P_model − P_mkt\| > 18% | `no_bet` + shrink aggressivo |
| Stato persistito | `data/processed/risk_state.json` | Audit + `python main.py predict` stampa alert se attivo |

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

---

## Dati Sackmann ATP / WTA

Fonte storica match, ranking e giocatori (Jeff Sackmann / Tennis Abstract, licenza CC BY-NC-SA 4.0).

> **Nota (2026):** i repo GitHub originali [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) e [JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta) risultano **404** (rimossi o non raggiungibili). Usare il **mirror archive**.

### Mirror consigliato

[Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive) — stessi CSV, snapshot aggiornato (ATP + WTA + slam point-by-point):

```
lib/
├── tennis-sackmann-archive-main/
│   ├── atp/                    ← atp_matches_*.csv, atp_players.csv
│   └── wta/                    ← wta_matches_*.csv, wta_players.csv
├── tennis_MatchChartingProject-master/
├── infotennis-main/
├── seeder-main/
├── tennis_atp-master/
├── TML-Database-master/
└── tennisgnn_predictions-main/
```

### Configurazione `.env`

**Opzione consigliata** — una variabile per entrambi i tour (path relativi alla root del progetto):

```env
SACKMANN_ARCHIVE_PATH=lib/tennis-sackmann-archive-main
MCP_CHARTING_PATH=lib/tennis_MatchChartingProject-master
INFOTENNIS_PATH=lib/infotennis-main
SEEDER_PATH=lib/seeder-main
```

Il modulo `sackmann.py` risolve automaticamente `{ARCHIVE}/atp` e `{ARCHIVE}/wta`.  
Se `.env` è vuoto, i moduli usano gli stessi default sotto `lib/` (`modules/lib_paths.py`).

**Opzione alternativa** — percorsi singoli (sovrascrivono l'archive):

```env
SACKMANN_ATP_PATH=lib/tennis-sackmann-archive-main/atp
SACKMANN_WTA_PATH=lib/tennis-sackmann-archive-main/wta
```

**Opzione cloud / CI** — download automatico se i CSV locali mancano:

```bash
python -c "from modules.data_update.sackmann import clone_sackmann_tour; print(clone_sackmann_tour(tour='wta'))"
```

Ordine di risoluzione in `sackmann.py`: `SACKMANN_ARCHIVE_PATH` → `SACKMANN_*_PATH` → `lib/` (archive, `tennis_atp-master`) → `data/raw/{atp|wta}/` → clone mirror GitHub.

### Setup Elo WTA

1. **Sackmann WTA** — da archive (cartella `wta/`) per Elo storico e settle pick
2. **Tennis Abstract WTA** (automatico): scaricato a ogni `predict` → `data/raw/tennis_abstract_elo_wta.json`
3. **Tour detection**: tornei con "Women's" / "WTA" usano engine WTA separato

Senza CSV WTA il sistema usa **Tennis Abstract + nomi charting** (fallback TA-only, predizioni possibili ma Elo meno ricchio).

---

## TML-Database (sorgente primaria ATP)

Modulo: `modules/data_update/tml.py`

| Priorità | Sorgente | Tour |
|----------|----------|------|
| 1 | `lib/TML-Database-master` o `data/raw/tml` (git pull) | ATP |
| 2 | Sackmann archive (`load_sackmann_matches`) | ATP gap-fill |
| — | Sackmann WTA | WTA only |

`load_tour_matches()` esegue merge per chiave `(data, winner, loser)`: TML vince sui duplicati.

```bash
python -c "from modules.data_update.tml import sync_tml; print(sync_tml())"
python main.py sync    # include sync TML
```

Env opzionale: `TML_PATH=lib/TML-Database-master`

---

## CLV live senza API a pagamento

Modulo: `modules/advisor/clv_live.py`

Cascade quote di chiusura (costo zero):

1. **Pinnacle guest API** — `modules/data_update/pinnacle_guest.py`
2. **Betfair LTP/back de-vigged** — proxy Pinnacle (r≈0.98 su tabelloni ATP/WTA)
3. **OddsPortal cache** — `data/raw/oddsportal_close.json` (job `.github/workflows/oddsportal-clv.yml`)
   Scraper Playwright: `python scripts/scrape_oddsportal_close.py` o `python main.py scrape-oddsportal`
   Per **Pinnacle** in geo IT/EU: imposta `ODDSPORTAL_PROXY` (proxy Malta/UK) nel `.env`
4. **tennis-data.co.uk** — storico PSW/PSL

`resolve_close_odds()` usato in `upcoming.py` → `advise.py`.  
`refresh_clv_close()` in `history.py` aggiorna pick pendenti prima del settle.

---

## Pipeline retrain + meta-learner

| Step | Comando / file |
|------|----------------|
| Feature store Parquet | `data/processed/features_v2.parquet` via `feature_store.py` |
| Retrain orchestrato | `python scripts/run_retrain_pipeline.py` o `python main.py retrain` |
| Meta-learner | `data/models/meta_learner.joblib` (+ alias `stacker.joblib`) |
| OOF temporale | `data/models/stacker_oof.joblib` — predizioni out-of-fold per fold |
| Calibrazione stacker | `data/models/calibration.json` → `brier_oof`, `brier_insample`, `cv: TimeSeriesSplit` |
| Fallback blend | 40/25/35 se meta-learner assente (`stacker.py`) |
| Cloud CI | `cloud-train.yml` → sync TML + `run_retrain_pipeline.py` |

**Anti-leakage stacker:** i dati sono ordinati strict per `tourney_date` (`mergesort`) prima del fit. Le predizioni OOF del meta-learner usano **solo** `TimeSeriesSplit(n_splits=5)` — ogni fold addestra solo su match storicamente antecedenti. Il modello finale viene rifittato sull'intero dataset ordinato per il deploy live.

```bash
python main.py retrain --min-year 2010
python main.py features --force
python main.py train   # XGBoost + stacker con TSCV
```

---

## Robustezza quantitativa

| Rischio | Mitigazione | Modulo |
|---------|-------------|--------|
| **Temporal leakage** (OOF stacker / XGBoost) | Sort per data + `TimeSeriesSplit`; Brier OOF ≠ in-sample | `stacker.py`, `train.py` |
| **MCP copertura asimmetrica** (Big Match vs early rounds) | Bayesian shrinkage verso media circuito se `<5` match chartati | `pressure.py` |
| **Entity resolution** (TML / Betfair / OddsPortal) | Registry SQLite con ID Sackmann/TML + alias; fuzzy solo come fallback | `player_registry.py`, `entity_resolution.py` |
| **Steam su dropping odds** | EV calcolato sempre sulla **quota corrente**; scarto se lo steam ha eroso il margine | `advise.py`, `playability.py` |

### Player registry (SQLite)

Path: `data/processed/player_registry.sqlite`

```bash
python main.py sync   # include sync Sackmann players + TML link + registry
```

Ordine risoluzione nome in `resolve_name()`:

1. **Registry SQLite** (`lookup_player_id` → `canonical_name`)
2. Alias JSON (`player_aliases.json`)
3. Fuzzy match runtime (`rapidfuzz`)

### Filtro steam (dropping odds)

In `advise.py`, se il dropping è **allineato** al pick ma:

- `EV(quota_corrente) < MIN_EDGE`, oppure
- il margine è calato >45% rispetto all'open (`erosion_ratio = 0.55`)

→ `action: "no_bet"` con motivo `steam: ...`. In `playability.py` il componente dropping scende a **0.15** (`steam_eroded: true`).

---

## Tennis-Data.co.uk (quote + risultati)

Portale [tennis-data.co.uk](http://www.tennis-data.co.uk/) di Joseph Buchdahl — **dati gratuiti** (CSV/Excel).

Modulo: `modules/data_update/tennis_data_portal.py`

| Tipo dati | URL remoto | Cache locale |
|-----------|------------|--------------|
| Stagione ATP | `/{year}/{year}.xlsx` | `data/raw/odds/{year}.csv` |
| Torneo ATP | `/{year}/{slug}.csv` | `data/raw/odds/tournaments/` |
| Torneo WTA | `/{year}w/{slug}.csv` | `data/raw/odds/wta/{slug}_{year}.csv` |
| Livescore | [livescore.tennis-data.co.uk](http://livescore.tennis-data.co.uk/) | `data/raw/tennis_livescore.json` |

**Sync automatico** (cache 7 gg stagioni, 3 gg tornei):

```bash
python main.py sync          # include tennis-data + livescore
python -c "from modules.data_update.tennis_data_portal import sync_tennis_data_portal; print(sync_tennis_data_portal())"
```

Tornei prioritari scaricati: Grand Slam + Masters 1000, **ATP e WTA**.

**Integrazione analisi:**
- Colonne **PSW/PSL** (Pinnacle) → CLV via `lookup_pinnacle_odds()` + `pinnacle_clv.py`
- Merge storico in training/backtest (`load_odds_all()`)
- A ogni `predict`, sync leggero se cache scaduta

**Livescore:** widget LiveXscores — parsing best-effort; può fallire se Cloudflare blocca i bot.

---

## Pipeline di analisi (live)

Comando principale: `python main.py predict` (o pulsante **Aggiorna calendario** in Streamlit, porta 8502).

### 1. Ingestion dati e segnali mercato

| Step | Modulo | Output |
|------|--------|--------|
| Match storici Sackmann ATP | `sackmann.py` | Elo engine ATP (`SACKMANN_ARCHIVE_PATH/atp` o `data/raw/atp/`) |
| Match storici Sackmann WTA | `sackmann.py` | Elo engine WTA (`SACKMANN_ARCHIVE_PATH/wta` o `data/raw/wta/`) |
| Elo Tennis Abstract | `tennis_abstract.py` | ATP + WTA cache JSON |
| Quote live | `betfair.py` | `data/raw/betfair_odds.json` |
| **Tennis-Data.co.uk** | `tennis_data_portal.py` | Quote ATP stagioni + tornei ATP/WTA (PSW/PSL Pinnacle) |
| **Livescore** | `tennis_livescore.py` | `livescore.tennis-data.co.uk` → cache JSON |
| Moneyway volume | `market_signals.py` → [Arbworld 1x2](https://arbworld.net/moneyway/tennis/1x2) | `arbworld_moneyway.json` |
| Dropping odds | `market_signals.py` → [OddsSafari sport 30](https://www.oddssafari.com/dropping-odds/sports/30) | `oddssafari_dropping.json` |
| OddsPortal close | `oddsportal_scraper.py` (Playwright) | `data/raw/oddsportal_close.json` — job `oddsportal-clv.yml` |
| Entity resolution | `entity_resolution.py` + **`player_registry.py`** | Registry SQLite (ID ATP/WTA) → alias JSON → fuzzy |

Cache segnali mercato: **30 min**. Betfair: **1 h** (o cache se API non disponibile).

### 2. Predizione per ogni match

Per ogni evento Betfair (fallback: match recenti Sackmann con quote storiche):

1. **Tour detection** — `Women's US Open` → WTA, `Men's US Open` → ATP
2. **Risoluzione giocatori** — registry SQLite → alias JSON → fuzzy (`H Dart` → `Harriet Dart`)
3. **Elo + CPI** — `blended_with_cpi()` modula peso superficie per velocità campo
4. **Markov dinamico** — BP save + tiebreak clutch (MCP) via `pressure.py`; shrinkage se `<5` match chartati
5. **ML live** — `live_features.py` → XGBoost con fatica/viaggio/hold-break
6. **Stacking** — meta-learner logistico su **OOF temporale** (`TimeSeriesSplit`), Brier OOF

Output: `p_win_a`, componenti, `cpi_norm`, `pressure_used`, `tour`, `model_low_confidence`.

### Precisione avanzata (dettaglio)

Vedi sezione [Layer predittivo](#layer-preditivo-3-livelli--meta-learner) e moduli:

| Miglioramento | Modulo |
|---------------|--------|
| Markov clutch/BP + shrinkage MCP | `modules/markov/pressure.py` |
| Elo × CPI | `modules/feature_engineering/elo.py` |
| Feature viaggio/hold-break | `modules/feature_engineering/features.py`, `live_features.py` |
| Meta-learner OOF temporale | `modules/model_training/stacker.py` |
| Player registry SQLite | `modules/data_update/player_registry.py` |
| CLV Pinnacle | `modules/advisor/pinnacle_clv.py`, `clv_live.py` |

### 3. Value bet (consiglio scommessa)

Modulo `advise.py` + `value.py` + `market_calibration.py`:

1. **Calibrazione** Isotonic/Platt su P stacker (se artifact presente)
2. **Shrink Bayesiano** verso mercato de-vigged: `P = w·P_cal + (1−w)·P_mkt`
3. **De-vig Shin** sulle quote → probabilità di mercato fair (`mkt_prob`)
4. **EV** per entrambi i lati: `EV = P × quota − 1` (es. 0.15 = **+15%**)
5. **Edge** in punti percentuali: `edge = P − mkt_prob`
6. **Kelly frazionato** (γ=0.20, cap dinamico per livello)
7. Scelta del lato con **Kelly-adjusted / Sharpe** più alto (non EV grezzo)

**Filtri no-bet** (bloccano la raccomandazione):

| Filtro | Soglia |
|--------|--------|
| EV minimo | ≥ 2.5% (`MIN_EDGE`) — calcolato sulla **quota corrente** |
| Probabilità minima | ≥ 38% (`MIN_PROB_PLAY`) |
| EV sanity hard | >30% (quote ≤3) / >25% (quote lunghe) → scarto |
| EV review | >20% e ≤ hard cap → `action: review` (no Telegram) |
| Divergenza mkt | \|P_model − P_mkt\| > 18% |
| Modello incerto | giocatore/i non identificati nel database |
| Artefatto 50/50 | `P ≈ 50%` e `P_elo ≈ 50%` senza ML |
| **Steam eroso** | dropping allineato al pick ma EV corrente sotto soglia o margine eroso >45% vs open |

Se passa i filtri → `action: "bet"` + `recommended` (pick, quota, `ev`, `ev_pct`, Kelly, `odds_sharpe`, `kelly_adj_rank`).

Campi EV nel JSON:
- `ev` — decimale (0.152 = 15.2% di rendimento atteso per unità puntata)
- `ev_pct` — stesso valore in percentuale (+15.2)
- UI Streamlit: colonna **EV % numerica** (`NumberColumn`) per ordinamento corretto (evita "+2%" > "+10%" lessicografico)
- Telegram mostra sempre **EV %** (es. `+15.2%`)

### 4. Indice giocabilità (0–100)

Modulo `playability.py`. Calcolato **dopo** il value bet, arricchisce ogni predizione.

**Formula composita** (pesi ridistribuiti se Moneyway/Drop assenti — **non** usare 0.50 neutro come se fosse un segnale reale):

```
Giocabilità = 100 × (
    w_value × value          # default 0.28; include penalità varianza quota
  + w_agree × model_agreement  # 0.16
  + w_kelly × kelly            # 0.16; Kelly-adjusted
  + w_market × market_quality  # 0.15
  + w_mw × moneyway            # 0.13; se missing → peso a value/kelly
  + w_drop × dropping_odds     # 0.12; se missing → peso a value/kelly
)
```

**Penalità finali:**
- Se `action = "review"` → score max **72** (sotto soglia alert)
- Se `action ≠ "bet"` (e non review) → score max **55**
- Se `EV < MIN_EDGE` → score max **45**
- Se quota ≥ 4.0 → score max **78**

**Bande:**

| Score | Band | Significato |
|-------|------|-------------|
| 0–30 | No bet | Non giocabile |
| 30–60 | Lean | Interessante ma debole |
| 60–75 | Playable | Giocabile con cautela |
| 75–90 | Strong | Alert Telegram ✅ |
| 90–100 | Premium | Massima convinzione |

**Soglia alert Telegram:** `MIN_PLAY_ALERT = 75` (Strong+) e solo `action=bet`.

---

## Dettaglio componenti giocabilità

Ogni campo è normalizzato 0–1 prima di applicare il peso. Valori alti **alzano** la giocabilità; valori bassi la **abbassano**.

### 1. Value (peso ~28%)

Misura edge **sostenibile**, non EV grezzo su quote lunghe.

| Input | Effetto |
|-------|---------|
| **EV** | `(EV − 2.5%) / 12%`, clamp 0–1 |
| **Edge vs mercato** | `edge / 8%`, clamp 0–1 |
| **Sharpe-like** | `EV / sqrt((odds−1)/ref)` |
| **Sostenibilità quota** | alto su ~1.8–2.0; basso su 4.90+ |
| Combinazione | mix EV + edge + Sharpe × sustainability |
| EV sotto 2.5% | componente bassa; cap score a 45 |

### 2. Model agreement (peso ~16%)

Accordo tra Markov, Elo e ML rispetto al blend finale.

| Input | Effetto |
|-------|---------|
| Modelli allineati (div < 15 pp) | → 1.0 |
| Divergenza media 15 pp | → 0.0 |
| ML assente | usa solo Markov + Elo |
| Modello ~50/50 senza ML | spesso blocca il bet prima (filtro incertezza) |

### 3. Kelly (peso ~16%)

Stake consigliato rispetto al cap, modulato da sostenibilità quota.

| Input | Effetto |
|-------|---------|
| Kelly = cap | → 1.0 |
| Kelly = 0 | → 0.0 |
| Kelly-adjusted | `kelly × (0.35 + 0.65·min(1, ref/odds))` |
| Formula base | mix `kelly/cap` + rank adjusted |

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
| **Assente / no match** | score **0.32** + peso ridistribuito (non 0.50 neutro) |
| Combinazione | 70% volume% + 30% liquidità; poi decay temporale T-12h→T-15min |

Interpretazione: favorito con >70% volume = segnale sharp; underdog con poco volume ma EV alto non viene penalizzato eccessivamente. **Segnale mancante non conta come neutro.**

### 6. Dropping odds (peso 12%) — OddsSafari

Movimento quote verso il nostro pick. **Non seguire ciecamente il dropping**: l'edge si valuta sulla quota **corrente**, non sull'open.

| Input | Effetto |
|-------|---------|
| Drop ≥10% allineato al pick | score alto (~0.55+) |
| Drop contro pick | penalità |
| Steam eroso | **0.15** + spesso `no_bet` in advise |
| **Assente / no match** | score **0.32** + peso ridistribuito |
| Combinazione | 70% volume pick + 30% liquidità |
| Match non trovato su Arbworld | neutro **0.5** |

Interpretazione: favorito con >70% volume = segnale sharp; underdog con poco volume ma EV alto non viene penalizzato eccessivamente.

### 6. Dropping odds (peso 12%) — OddsSafari

Movimento quote verso il nostro pick. **Non seguire ciecamente il dropping**: l'edge si valuta sulla quota **corrente**, non sull'open.

| Scenario | Score |
|----------|-------|
| **Steam eroso** (EV corrente sotto soglia o margine eroso vs open) | **0.15** — pick bloccato anche in `advise.py` |
| Drop ≥10% **allineato** al pick (margine ancora valido) | 0.55 + drop/40 (max ~1.0) |
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

### Apprendimento online automatico (GitHub Actions)

Workflow: `.github/workflows/auto-learn.yml` — cron **04:00 e 16:00 UTC** + `workflow_dispatch`.

| Step | Comando | Effetto |
|------|---------|---------|
| Sync risultati | `python main.py sync` | TML + Sackmann ATP/WTA aggiornati |
| Settle + learn | `python scripts/auto_learn_cloud.py --notify` | Chiude pick, aggiorna `calibration.json` |
| Audit BCR | `python main.py metrics` | `live_metrics.json` |
| Git push | commit `[skip ci]` | Persiste SQLite + JSON tra le run |

**Cosa impara il sistema** (≥12 pick chiuse, in `calibration.json` → `online_learn`):

| Parametro appreso | Applica a |
|-------------------|-----------|
| `min_edge_suggested` | Soglia EV in `predict` (via `effective_min_edge()`) |
| `alert_min_suggested` | Soglia Telegram Strong (75 vs 80) |
| `dropping_boost` / `moneyway_boost` | Giocabilità (`learned_playability_adjustment`) |
| Penalità bande Lean/Playable | Se hit rate storico basso |
| BCR Pinnacle <52% (n≥15) | Alza `min_edge_suggested` a 3.5% |

Requisiti GitHub: **Settings → Actions → Workflow permissions → Read and write**.  
Segreti opzionali: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (riepilogo post-learn).

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

`settle_from_results()` confronta pick pendenti con risultati a cascata (ultimi 14 gg):

1. **TML** (ATP primario, `git pull` prima del settle)
2. **Sackmann** (WTA + gap-fill ATP)
3. **Betfair settled** (mercati MATCH_ODDS chiusi con runner WINNER)
4. **FlashScore / diretta.it** (feed ninja, best-effort)
5. **tennis-data.co.uk** (CSV Winner/Loser storici)

Ogni pick chiusa registra `settle_source` in `our_history.sqlite`.

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

## Fase live — paper trading / shadow period

**Regola:** non modificare struttura modello (stacker/XGBoost/Markov) per i prossimi **200–300 match**. Monitorare solo metriche di esecuzione.

### KPI primario: BCR Pinnacle

| Metrica | Target | Fonte |
|---------|--------|-------|
| **Beat Closing Rate** (quota bet > chiusura Pinnacle de-vigged) | **> 55%** | `our_history.sqlite` → `beat_close` + `close_source` Pinnacle |

ROI sui primi 100 bet è **varianza** — il BCR conferma edge matematico vs mercato sharp.

### Finestra validazione live (FREEZE attivo)

**Priorità assoluta:** accumulare **200–300 match** settle con chiusura Pinnacle senza toccare l'architettura.

| Regola | Stato |
|--------|--------|
| Online learn → `calibration.json` | **BLOCCATO** (solo report in `online_learn_report.json`) |
| Retrain ML (CI `cloud-train`, `weekly-train`) | **BLOCCATO** |
| Boost giocabilità appresi (moneyway/dropping) | **BLOCCATI** |
| Settle pick + BCR audit | **ATTIVO** |
| Predict + Telegram | **ATTIVO** |
| Circuit breaker drawdown (capitale) | **ATTIVO** (non è learning) |

Config: `data/processed/validation_freeze.json` — si disattiva automaticamente a **200+ pick Pinnacle settle** (`active: false`); override manuale con `LIVE_VALIDATION_FREEZE=0` o `=1`.

```bash
python main.py metrics          # BCR + avanzamento finestra → live_metrics.json
python main.py learn            # settle only (no weight updates)
python main.py predict --metrics
```

Report: `data/processed/live_metrics.json`

### Audit slippage Telegram

Tabella `alert_log` in `our_history.sqlite` — ogni alert registra `odds_at_alert`, `ev_at_alert`.

- Snapshot **T+3 min** su cache Betfair (`refresh_slippage_snapshots`)
- **Steam entro 3 min:** quota pick cala >3% → flag `steam_within_3m`
- Se steam >40% degli alert: valutare cron Betfair ogni **10–15 min** pre-match

### Transizione superficie (solo inferenza live)

Modulo `surface_transition.py` — primi **7 giorni** dopo cambio stagione (Clay giu, Grass giu, Hard ago):

- Moltiplicatore peso Elo surface: **0.50 → 1.00** (più peso rating global)
- Metadato `surface_transition` nelle predizioni Betfair
- **Non** modifica training/backtest Elo engine

---

## Esecuzione rapida

```bash
# Analisi completa live
python main.py predict

# Analisi + invio Telegram (se credenziali in .env)
python main.py predict --notify

# Chiudi pick pendenti + apprendimento + BCR
python main.py learn

# Audit fase live (BCR Pinnacle + slippage)
python main.py metrics

# Verifica dati Sackmann (ATP + WTA)
python -c "from modules.data_update.sackmann import sync_sackmann_atp, sync_sackmann_wta; print(sync_sackmann_atp()); print(sync_sackmann_wta())"

# Training completo (features + XGBoost + stacker)
python main.py features && python main.py train

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
| **OddsSafari Dropping** | [oddssafari.com/dropping-odds/sports/30](https://www.oddssafari.com/dropping-odds/sports/30) | ✅ Sì | **Integrato** | Quote in calo — componente **dropping_odds** (12%). Bonus solo se margine ancora valido sulla quota corrente; steam eroso → no-bet |
| **OddsPortal** | [oddsportal.com/tennis](https://www.oddsportal.com/tennis/) | ⚠️ Utile | **Parziale** | Scraper Playwright → `oddsportal_close.json` per CLV close; geo IT può richiedere `ODDSPORTAL_PROXY` |

### OddsPortal — integrazione parziale (CLV close)

- Scraper **Playwright** attivo: `python main.py scrape-oddsportal` → `data/raw/oddsportal_close.json`
- Usato in cascade CLV (`clv_live.py`) come fallback dopo Pinnacle guest e Betfair LTP
- Limiti: geo IT reindirizza a `centroquote.it`; Pinnacle spesso assente senza `ODDSPORTAL_PROXY`
- **Non ancora integrato:** confronto multi-book live in giocabilità / value bet

---

## Upgrade quant 10/10 (2026)

### 1. Serve-Elo / Return-Elo (`modules/feature_engineering/serve_return_elo.py`)

| Componente | Dettaglio |
|------------|-----------|
| Aggiornamento | Da `w_svpt`, `w_1stWon`, `w_2ndWon` (e simmetrico loser) |
| Pre-match | `ServeReturnEloEngine.pre_match_ratings()` → serve/return per superficie |
| Markov | `estimate_serve_probs(..., serve_elo_a, return_elo_b, cpi_factor)` in `chain.py` |
| Live | `TourBundle.sr_elo_engine` in `upcoming.py` → `predict_match()` |

### 2. CPI dinamico + densità aria (`air_density.py`, `cpi.effective_cpi`)

```
ρ = (P − 0.378·e) / (R·T)     # temperatura, umidità, pressione/altitudine
CPI_eff = CPI_nom × (ρ_ref / ρ)^0.22
```

- Input: Open-Meteo (`weather.py`) + altitudine venue (`altitude.py`)
- Effetto: sessioni calde/notturne ad alta quota (Madrid, Roma) modulano velocità campo in tempo reale

### 3. Decadimento temporale segnali mercato (`market_timing.py`)

| Timing | Peso relativo |
|--------|---------------|
| T−15 min | **1.0×** (riferimento) |
| T−12 ore | **~0.33×** (rapporto 3:1) |

Formula: `exp(−λ · (minuti − 15))` con λ calibrato su ratio 3×. Applicato a componenti **moneyway** (13%) e **dropping** (12%) in `playability.py`.

### 4. Sotto-modello retirement (`retirement_risk.py`)

Feature: età (DOB Sackmann), fatica 72h, rest days, storico RET/DEF, favorito acciaccato.

```
P(ritiro) → penalità EV/Kelly × rule_penalty[bookmaker]
```

| Book | Regola | Penalità stake |
|------|--------|----------------|
| bet365 | 1-ball | 0.55 |
| pinnacle | 1-set | 0.25 |
| default | 1-set | 0.30 |

Integrato in `advise()` via `retirement_context`.

### 5. Graph entity resolution (`player_graph.py`)

Ordine risoluzione nomi in `resolve_name()`:

1. **Graph match** — opponent + data torneo su `match_edges` (SQLite)
2. Registry SQLite (`player_registry.py`) — ID Sackmann/TML + alias
3. Vincoli IOC / birth_year
4. Fuzzy `rapidfuzz` (solo fallback)

Popolamento automatico in `build_upcoming()`: biographics da `*_players.csv` + ultimi 8000 match come edge.

```bash
python -c "from modules.data_update.player_graph import graph_stats; print(graph_stats())"
```
