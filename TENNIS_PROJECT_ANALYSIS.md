# TENNIS PROJECT ANALYSIS — Diagnosi Quant & Piano di Sblocco

**Ruolo:** Senior Quant / Betting Systems  
**Data:** 2026-09-29  
**Scope:** codebase `tennis-predictor` (moduli predizione, quote, filtri, storico, output Telegram/UI)  
**Problema dichiarato:** il sistema non consiglia quasi mai, oppure non trova valore.

---

## Executive summary

Il modello **produce probabilità e EV** su centinaia di match; il collo di bottiglia non è “assenza di analisi”, ma una **catena di filtri in serie** che, nello stato attuale, riduce l’output a quasi zero.

Snapshot empirico su `data/processed/upcoming_predictions.json` (173 match):

| Stato | N |
|-------|---|
| `action=no_bet` | 159 |
| `action=shadow` | 12 |
| `action=bet` | **2** |
| `bet` + giocabilità ≥60 (Telegram) | **0** |

Cause dominanti (in ordine di impatto operativo):

1. **Circuit breaker ATTIVO** → soglia EV effettiva **7%** (non 5%). Drawdown corrente ≈ **24.6%** > soglia 20% (`risk_state.json`).
2. **Shrink Bayesiano verso il mercato** che comprime l’edge modello prima del calcolo EV.
3. Filtri hard: **quota 1.70–5.00**, **Kelly ≥ 0.3%**, **P ≥ 34%**, sanity EV, sharp consensus su soft book.
4. **Giocabilità** tipicamente 20–52 anche sui pochi `bet` → Telegram (soglia 60) resta muto.
5. **BCR Betfair quality a n≈0** (auth Betfair `STRONG_AUTH_CODE_REQUIRED` + chiusure solo fallback) → freeze validazione bloccato, shadow limitate, feedback loop spezzato.

Il sistema è architetturalmente solido (stack multi-layer + risk controls). È **over-constrained in esecuzione**, non “vuoto di segnale”.

---

## 1. Panoramica dell’architettura e dei flussi

### 1.1 Entry point

| Superficie | Ruolo |
|------------|--------|
| `main.py` | CLI: `sync`, `news`, `build`, `features`, `retrain`, `train`, `backtest`, `predict`, `metrics`, `learn`, `full` |
| `app.py` | Dashboard Streamlit (calendario, BCR, dettaglio pick) |
| `scripts/notify_cloud.py` | Predict + Telegram (CI `/30`) |
| `scripts/auto_learn_cloud.py` | Settle + online learn + BCR |
| `.github/workflows/` | `telegram-alerts.yml`, `auto-learn.yml`, `cloud-train.yml`, `weekly-train.yml`, `oddsportal-clv.yml` |

Flusso live tipico:

```
sync (dati storici/extra)
  → features / train (offline)
  → python main.py predict [--notify]
       upcoming.py: palinsesto Betfair + Kambi
       MatchPredictor: Elo + Markov + ML + stacker + calibrazione
       market_calibration: shrink verso mercato
       advise.py: EV / Kelly / filtri → action
       playability.py: score 0–100
       shadow_bet.py: promozioni shadow sotto freeze/CB
       injury_feed: hard-block withdrawal
       history.sqlite: archive paper/bet/shadow
       alerts.py: Telegram (solo bet + filtri + play≥60)
```

### 1.2 Ingestion dati

| Fonte | Modulo | Uso |
|-------|--------|-----|
| Sackmann ATP/WTA + archive | `data_update/sackmann.py`, `player_registry.py` | Storico match, ranking, Elo base |
| TML Database | `data_update/tml.py` | Merge/gap-fill ATP |
| Match Charting Project | `data_update/charting.py`, `markov/pressure.py` | BP save, clutch, serve profile |
| Tennis Abstract Elo | `data_update/tennis_abstract.py` | Prior esterno / consenso |
| CourtSpeed / UTS CPI | `cpi.py`, `courtspeed.py`, `uts.py` | Velocità campo |
| Tennis-data.co.uk odds | `tennis_data_odds.py` | Quote storiche train |
| Betfair Exchange | `betfair.py` | Quote live sharp + market_id (BCR) |
| Kambi/Unibet Guest | `kambi_unibet.py` | Copertura Challenger/ITF |
| Moneyway / Dropping | `market_signals.py` | Componenti giocabilità |
| Injury/news RSS | `injury_feed.py` | Hard-block withdrawal |
| Weather / altitude / air density | `weather.py`, `altitude.py`, `air_density.py` | Aggiustamento serve |
| RapidAPI / Sofa / ESPN | settle live risultati | Hit/CLV post-match |
| OddsPortal | `oddsportal_*` | Close secondario (non KPI BCR) |

### 1.3 Dal dato grezzo all’output

```
Raw CSV/API
  → dataset_loader + feature_engineering (Elo surface, fatigue, travel, CPI…)
  → model_training (XGBoost) + stacker OOF (logreg temporale)
  → MatchPredictor.predict_match (p_markov, p_elo, p_ml → p_blend → p_cal)
  → signal_consensus (forma/superficie/qualità/TA)
  → apply_bayesian_shrinkage (P_finale = w·P_model + (1−w)·P_mkt)
  → enrich_value + advise (Shin/Power, EV, Kelly, filtri)
  → enrich_playability
  → recommended / best_play / action
  → upcoming_predictions.json + our_history.sqlite
  → UI + Telegram
```

**Artefatti chiave**

- `data/processed/upcoming_predictions.json` — calendario + consigli
- `data/processed/our_history.sqlite` — pick archiviate (bet/shadow/paper) + settle/CLV
- `data/processed/live_metrics.json` — BCR / hit / governance
- `data/processed/risk_state.json` — circuit breaker
- `data/processed/validation_freeze.json` — freeze architettura
- `data/models/best_model.joblib`, `calibration.json` — ML + soglie apprese

### 1.4 Storico e KPI

- Archive: `modules/data_update/history.py` (`matches` table).
- KPI primario dichiarato: **BCR Betfair quality** (BSP / T−1 / T−5 / T−60) su `action∈{bet,shadow}`.
- Esclusi: `betfair_ltp_fallback` e close ≈ quota bet (`BCR_MIN_CLOSE_DELTA`).
- Secondario: BCR Kambi, hit-rate, ROI (solo dopo campione ampio).

---

## 2. Analisi del modello predittivo (perché non produce output)

### 2.1 Stack di probabilità

`modules/predictor/predict.py` — `MatchPredictor.predict_match`:

1. **Elo** (`feature_engineering/elo.py` + serve/return Elo):  
   \(P_{elo} = \sigma(R_a - R_b)\); blend global/surface con peso `ELO_SURFACE_WEIGHT=0.68`, modulato da CPI e transizione superficie.
2. **Markov** (`markov/pressure.py`): P(serve) da Serve-Elo × Return-Elo + clutch MCP (BP/TB) + weather/CPI/skills → P(match) best-of-3/5.
3. **ML XGBoost** (`model_training/train.py`): feature live (fatica 7/14g, viaggio, rest, H2H, hold/break, densità dati…).
4. **Meta-stacker** (`stacker.py`): pesi OOF temporali su (p_markov, p_elo, p_ml). Fallback se manca artifact: **40% Markov / 25% Elo / 35% ML**.
5. **Calibrazione** Isotonic/Platt (`prob_calibrator.py`) su P stacker.
6. **Consenso segnali** (`signal_consensus.py`):  
   `p_signal = 0.30·surface + 0.25·form + 0.20·quality + 0.15·TA + 0.10·stack` prima dello shrink.
7. **Shrink mercato** (`market_calibration.py`):  
   \(P = w\cdot P_{model} + (1-w)\cdot P_{mkt}^{Shin}\)  
   con \(w\) che scende su ITF, bassa densità, divergenza ≥12%.

**Implicazione quant:** dopo lo shrink, l’EV osservabile è tipicamente un **residuo piccolo** rispetto al mercato. Chiedere EV ≥ 5–7% *dopo* shrink su quote sharp è una richiesta molto aggressiva: equivale a richiedere che il modello batta il mercato di diversi punti di probabilità su un book già efficiente.

### 2.2 Value / staking

- De-vig default: **Shin** (`advisor/value.py`).
- \(EV = P_{finale}\times quota - 1\).
- Kelly frazionato \(\gamma=0.20\), cap per livello torneo (`KELLY_CAP_BY_LEVEL`).
- Ranking pick: **Kelly-adjusted / Sharpe-like**, non EV grezzo (`advise._rank_key`).

### 2.3 Catena di decisione `action`

Ordine effettivo in `advise.py` + post-process:

```
no_bet_reasons (EV / odds / Kelly / P)
  + model_uncertainty / ITF gate
  + steam_eroded
  + ev_sanity / market divergence
  + sharp_consensus
  → review se EV > 20% sotto hard-cap
  → retirement adjust Kelly
  → shadow_bet (se freeze/CB e fallisce bet)
  → injury hard-block
  → daily exposure Kelly scale
  → playability score
  → Telegram: action=bet AND play≥60 AND filtri quota/EV/Kelly
```

### 2.4 Soglie attuali (costanti)

Da `modules/constants.py` + stato runtime:

| Parametro | Valore | Effetto |
|-----------|--------|---------|
| `MIN_EDGE` | **5.0%** | Floor EV per bet |
| `CIRCUIT_BREAKER_MIN_EDGE` | **7.0%** | EV sotto stress |
| **Effettivo ora** | **7.0%** | CB attivo (`risk_state.json`) |
| `MIN_KELLY` | **0.3%** bankroll | Blocca edge sottili |
| `MIN_ODDS_PLAY` / `MAX` | **1.70 / 5.00** | Taglia favoriti corti e longshot |
| `MIN_PROB_PLAY` | **34%** | Blocca underdog “puri” |
| `EV_REVIEW_THRESHOLD` | 20% | `review` (no Telegram) |
| Sanity EV | >30% (odds≤3) / >25% (lunghe) | `no_bet` |
| `MKT_DIVERGENCE_MAX` | 20% | Hard no_bet |
| Steam | drop≥8% + erosione 40% | Hard no_bet |
| `MIN_PLAY_ALERT` | 60 (floor; online_learn può alzare a 80) | Telegram |
| Shadow | EV≥1.5%, Kelly=0 | Solo sample BCR |
| Freeze validazione | attivo, target 200–250 | **Non** blocca `bet`; blocca retrain / learn writes / boost playability |

**Nota Telegram vs CB:** in `alerts._passes_value_filters` l’EV è confrontato con `MIN_EDGE` (5%) **hardcoded**, non con l’edge effettivo del circuit breaker (7%). Un bet passato con min_edge=7% passa comunque il check Telegram sull’EV; il vero silenziatore Telegram resta `playability ≥ 60`.

### 2.5 Evidenza empirica — cosa blocca

Su 173 predizioni, frequenza motivi `no_bet` (best_play):

| Motivo (normalizzato) | Occorrenze (ordine) |
|-----------------------|---------------------|
| EV sotto soglia | ~98 |
| Probabilità &lt; 34% | ~69 |
| Kelly &lt; 0.3% | ~67 |
| EV sanity | ~51 |
| Modello incerto (entity resolve) | ~32 |
| Quota &lt; 1.70 o &gt; 5.00 | ~38 |
| Artefatto 50/50 | ~11 |
| Sharp consensus (Kambi odds&gt;3) | decine |

Tra i 55 match che *sulla carta* rispettano EV≥5% + quota 1.70–5 + Kelly≥0.003:

- solo **2** diventano `bet`
- **5** `shadow`
- **48** restano `no_bet` per **prob_low / sharp / sanity / edge effettivo 7%**

I 2 bet reali (Basilashvili, Shapovalov) hanno giocabilità **48.3** e **51.7** → **fuori Telegram**.

### 2.6 Doppio vincolo: “non consiglio” vs “non trovo valore”

Due fenomeni distinti:

**A. Over-filtering (output selettivo)**  
Il modello calcola EV, ma i gate lo trasformano in `no_bet`. Qui il problema è di **policy di esecuzione**, non di feature missing.

**B. Compressione del segnale (valore raro)**  
Shrink + book sharp → distribuzione EV centrata vicino a 0. Con CB a 7%, la coda destra utile è magrissima. Molti EV “alti” su Kambi/ITF vengono poi uccisi da **sanity** o **sharp consensus** (corretto come anti-falso-positivo, ma lascia il palinsesto vuoto).

---

## 3. Specificità del tennis: cosa è gestito e dove resta debole

### 3.1 Superfici — gestito bene

- Elo multisuperficie + peso surface 0.68 (`elo.py`, `serve_return_elo.py`).
- **CPI** torneo + weather/altitude → aggiustamento P(serve) (`cpi.py`, `air_density.py`).
- **Transizione superficie** primi 7 giorni stagione (`surface_transition.py`).
- Feature surface WR e livello torneo nel ML.

Valutazione: layered e coerente. Non è la causa del “secco”.

### 3.2 Stanchezza / calendario — gestito in feature, soft in decision

In `features.py` / `live_features.py`:

- `fatigue_minutes_7d` / `14d`
- `rest_days`
- `travel_km`, timezone shift (jet lag)

Entra in **ML** e in **P(ritiro)**; **non** c’è un hard-block “troppa fatica → no_bet” dedicato (a parte ritiro).  
Possibile miglioramento: gate soft su mismatch fatica estremo (es. BO5 dopo match lungo il giorno prima), ma non è il bottleneck attuale.

### 3.3 Ritiri — gestito pre-match, non live-betting

`retirement_risk.py`:

- P(ritiro) da età, fatica, rest, storico RET/DEF/WO, boost news.
- Cap `RETIREMENT_MAX_P=0.45`.
- Modula EV/Kelly con penalità per regola book (`1_ball` / `1_set`).
- `injury_feed.py`: withdrawal sul pick → **hard `no_bet`** + cap playability 35.

**Non** c’è motore live in-play (no trading mid-match). Scope = pre-match moneyline.

### 3.4 Tornei inferiori

- `ANALYZE_LOWER_TIERS=True`, partial resolve Betfair OK.
- Shrink ITF forte (`BAYES_SHRINK_W_ITF=0.22`) + governance BCR ITF.
- Kambi copre ITF/Challenger; sharp consensus spesso **blocca** longshot Kambi senza conferma Betfair.

### 3.5 Infra Betfair / BCR (blocco strutturale del feedback)

- Login locale: `STRONG_AUTH_CODE_REQUIRED` → niente book live affidabile.
- Senza snapshot T−60/T−5/T−1 / BSP, le chiusure restano `betfair_ltp_fallback` → **escluse dal KPI**.
- Freeze validazione: target 200–250 settle Betfair quality; progresso ~**0** → freeze resta ON.
- Shadow esistono (12 sul calendario) ma senza close quality il BCR non sale → stallo strategico.

Questo non riduce direttamente i consigli UI, ma **impedisce di rilassare i filtri in modo data-driven** (online_learn / unfreeze).

---

## 4. Proposta di sblocco e ottimizzazione

Obiettivo: ripristinare un flusso controllato di consigli (ordine 5–15 pick/giorno su palinsesto pieno) senza riaprire i falsi positivi ITF da EV 20%+ su quote 10+.

### 4.1 Interventi immediati (alto impatto / basso rischio codice)

#### A. Circuit breaker — reset o policy meno cieca

Stato: drawdown 24.6%, `min_edge` forzato a **7%**.

Opzioni:

1. **Reset bankroll paper** / ricalcolo CB solo su pick post-filtri nuovi (evita che lo storico “sporco” di settembre tenga il breaker forever).
2. Oppure alzare `DRAWDOWN_BREAKER_PCT` a **0.30** e/o far sì che il CB alzi lo stake-cap invece dell’EV floor (meglio: stress = Kelly×0.5, non EV 7%).

File: `modules/constants.py`, `modules/advisor/risk_controls.py`, `data/processed/risk_state.json`.

#### B. Allentare il trio EV / odds / Kelly (coerente su bet + Telegram + playability)

Proposta “rischio controllato” (sostituisce il pacchetto 5% / 1.70–5 / 0.3%):

| Parametro | Oggi | Proposta fase 1 | Proposta fase 2 (se BCR quality n≥30) |
|-----------|------|-----------------|--------------------------------------|
| `MIN_EDGE` | 5% | **3.0%** | 2.5% |
| `CIRCUIT_BREAKER_MIN_EDGE` | 7% | **4.5%** | 3.5% |
| `MIN_ODDS_PLAY` | 1.70 | **1.55** | 1.50 |
| `MAX_ODDS_PLAY` | 5.00 | **4.50** (più stretto in alto) | 4.00 |
| `MIN_KELLY` | 0.003 | **0.0015** (0.15%) | 0.001 |
| `MIN_PROB_PLAY` | 34% | tenere 34% su odds≤3.2; su 3.2–4.5 richiedere P≥38% | — |

Razionale:

- EV 3% post-shrink è già edge “serio” su tennis sharp.
- Quota max **più bassa** (4.5) riduce falsi positivi longshot meglio di un EV floor altissimo.
- Kelly 0.15% evita di scartare edge buoni su cap Challenger bassi.

File: `modules/constants.py` (+ mirror brief / caption `app.py`).

#### C. Telegram / giocabilità — disaccoppiare alert da score composito

Oggi: anche i bet “buoni” stanno a play≈50 → 0 alert.

Proposte:

1. Alert se `action=bet` **e** (play≥**50** **oppure** `odds_sharpe` sopra soglia), **oppure**
2. Cap playability meno punitivo: se `action=bet` e passa filtri EV/odds/Kelly, **floor playability = 62** (force-pass alert). I segnali MW/Drop restano informativi ma non silenziano il canale.

File: `modules/notify/alerts.py`, `modules/advisor/playability.py` (`MIN_PLAY_ALERT`).

#### D. Sharp consensus — solo su soft book, non doppio kill

Su Kambi con odds∈[3, 4.5]: invece di hard-block, promuovere a `review` o richiedere solo `|P − P_mkt| < 12%`.  
Mantenere hard-block per odds &gt; 5 (già fuori fascia).

File: `modules/advisor/market_calibration.py` → `sharp_consensus_reasons`.

### 4.2 Interventi medi (qualità segnale, non solo volume)

1. **EV post-shrink report**: loggare `ev_pre_shrink` vs `ev_post_shrink` in predizione — diagnostica se lo shrink azzera sistematicamente la coda.
2. **Sanity EV legata a densità**: su ATP/WTA Masters con densità ≥50, alzare soft-review e ridurre hard-sanity; su ITF tenere strict.
3. **Entity resolution**: 32 blocchi “modello incerto” — migliorare alias registry riduce no_bet “tecnici” senza toccare edge.
4. **Betfair auth**: risolvere `STRONG_AUTH` (cert client o sessione 2FA) altrimenti BCR e unfreeze restano impossibili; senza di ciò ogni tuning filtri è cieco.

### 4.3 Cosa NON allentare (anti falso positivo)

- Hard sanity su EV &gt; 25–30% su ITF/Kambi.
- News withdrawal hard-block.
- Steam erosion aggressivo (drop≥8%).
- Divergenza modello/mercato &gt; 20% pre-shrink.
- Inclusioni di quote &gt; 6–8 senza sharp confirm.

### 4.4 Sequenza operativa consigliata

```
1. Fix Betfair login (STRONG_AUTH) → snapshot prematch → BCR quality n>0
2. Reset/ricalibra circuit breaker (non tenere EV@7% su drawdown paper sporco)
3. Patch costanti fase 1 (EV 3%, odds 1.55–4.50, Kelly 0.15%)
4. Floor playability su action=bet per Telegram
5. Misura 7 giorni: n_bet/giorno, Telegram sent, BCR quality, hit-rate shadow
6. Solo poi fase 2 o online_learn min_edge
```

### 4.5 Target di accettazione (7 giorni)

| Metrica | Target sano |
|---------|-------------|
| `action=bet` / giorno | 3–12 (palinsesto misto) |
| Telegram value msg / giorno | ≥1 se esiste ≥1 bet |
| % bet con odds 1.55–3.5 | ≥70% del volume |
| BCR Betfair quality n | crescente (≥1 close/giorno) |
| Hit-rate shadow (indicativo) | non peggio del mercato di &gt;5pp su n≥20 |

---

## 5. Mappa file → responsabilità

| Area | Path |
|------|------|
| Costanti / soglie | `modules/constants.py` |
| Predizione | `modules/predictor/predict.py` |
| Elo / fatica / travel | `modules/feature_engineering/*` |
| Markov / pressure | `modules/markov/*` |
| Shrink / sharp / divergenza | `modules/advisor/market_calibration.py` |
| Bet/no-bet | `modules/advisor/advise.py`, `staking.py` |
| Giocabilità | `modules/advisor/playability.py` |
| Risk / CB | `modules/advisor/risk_controls.py` |
| Shadow | `modules/advisor/shadow_bet.py` |
| Ritiro | `modules/advisor/retirement_risk.py` |
| Palinsesto | `modules/data_update/upcoming.py` |
| Betfair / BCR close | `modules/data_update/betfair.py`, `history.py` |
| Telegram | `modules/notify/alerts.py` |
| Freeze | `modules/advisor/validation_freeze.py` |
| UI | `app.py` |

---

## 6. Conclusione

Il tennis-predictor **non è “muto” perché il modello non calcola**: è muto perché:

1. il **circuit breaker** ha portato l’asticella a **EV ≥ 7%** post-shrink;
2. i filtri **quota / Kelly / P / sharp / sanity** tagliano la coda rimanente;
3. la **giocabilità &lt; 60** silenzia Telegram anche sui rarissimi bet;
4. il **loop BCR Betfair** è rotto (auth + close quality), quindi non si può rilassare in modo scientifico.

La via corretta non è “togliere i filtri”, ma **ribilanciare esecuzione** (EV 3%, banda quote più corta in alto, Kelly più basso, alert disaccoppiato) **dopo** aver ripristinato Betfair e spezzato il deadlock del circuit breaker.

---

## Appendix — Inventario artefatti modello

| File | Ruolo |
|------|-------|
| `data/models/best_model.joblib` | XGBoost |
| `data/models/stacker.joblib` / `meta_learner.joblib` | Meta-learner |
| `data/models/calibration.json` | Calibrazione + `online_learn` |
| `data/models/online_learn_report.json` | Report learn (anche sotto freeze) |
| `data/processed/telegram_alerts_sent.json` | Dedup alert 21 gg |
| `data/processed/validation_freeze.json` | Stato freeze BCR |

---

### Aggiornamento operativo — Fase 1 unlock (2026-09-29)

Implementato in codice (vedi `modules/constants.py`, `risk_controls.py`, `alerts.py`, `playability.py`, `market_calibration.py`):

| Parametro | Prima | Fase 1 |
|-----------|-------|--------|
| `MIN_EDGE` | 5% | **3%** |
| CB stress EV | 7% | **4.5%** + **Kelly ×0.5** |
| CB metriche | storico intero | da **2026-09-29** (`CIRCUIT_BREAKER_METRICS_FROM`) |
| Odds | 1.70–5.00 | **1.55–4.50** |
| `MIN_KELLY` | 0.3% | **0.15%** |
| Telegram | play ≥60 | **bet + filtri core** (floor play 50/55) |
| Sharp soft 3–4.5 | hard-block | soft se divergenza ≤12% |

*Documento generato da analisi statica del codice + audit su `upcoming_predictions.json` / `risk_state.json` (2026-09-29), integrato con briefing esplorativo completo del repo. Fase 1 applicata sul codice; rieseguire `python main.py predict` per rigenerare il calendario.*
