"""Dashboard Streamlit — Tennis Value Betting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Tennis Predictor", page_icon="🎾", layout="wide")

st.title("🎾 Tennis Predictor — Value Betting")
st.caption("Elo multisuperficie · Markov punto→match · ML · Shin de-vig · Kelly frazionario")

tab_cal, tab_back, tab_data, tab_elo = st.tabs([
    "Calendario & Value", "Backtest", "Dati & Pipeline", "Elo Rankings"
])

with tab_cal:
    pred_path = ROOT / "data" / "processed" / "upcoming_predictions.json"
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        if st.button("Aggiorna calendario", type="primary"):
            with st.spinner("Scarico quote Betfair, calcolo predizioni e alert Telegram..."):
                import subprocess
                subprocess.run(
                    [sys.executable, "main.py", "predict", "--notify"],
                    cwd=ROOT,
                    check=False,
                )
            st.rerun()

    if pred_path.exists():
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        from modules.data_update.calendar_utils import normalize_predictions_calendar

        preds = normalize_predictions_calendar(preds)
    else:
        preds = []

    if not preds:
        st.warning(
            "Nessun match in calendario. Clicca **Aggiorna calendario** oppure esegui "
            "`python main.py predict` (serve modello trainato + credenziali Betfair nel .env)."
        )
    else:
        bets = [p for p in preds if p.get("action") == "bet"]
        playable = [p for p in preds if float(p.get("playability") or 0) >= 60]
        alertable = [p for p in preds if float(p.get("playability") or 0) >= 75]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Match analizzati", len(preds))
        c2.metric("Value bet (EV+)", len(bets))
        c3.metric("Giocabili ≥60", len(playable))
        c4.metric("Alert ≥75", len(alertable))
        c5.metric("Fonte quote", preds[0].get("odds_source", "—") if preds else "—")
        st.caption(
            "Calendario da **oggi** (Europe/Rome). Clicca **Aggiorna calendario** dopo rinvii meteo "
            "per sincronizzare orari Betfair."
        )
        st.caption(
            "Elo multisuperficie · Markov · ML calibrato (Isotonic/Platt) · "
            "prior mercato Bayesiano · Kelly/Sharpe ranking · Shin de-vig"
        )
        st.caption(
            "**Giocabilità 0–100**: value/EV penalizzato da varianza quota, accordo modelli, "
            "Kelly-adjusted, qualità mercato, Moneyway e dropping (assenti ≠ neutro 0.50). "
            "Alert Telegram solo `action=bet` e ≥75. EV >20% → review; EV >25–30% → scarto."
        )

        rows = []
        for p in preds:
            rec = p.get("recommended") or {}
            ev = rec.get("ev")
            ev_pct = rec.get("ev_pct")
            if ev_pct is None and ev is not None:
                ev_pct = round(float(ev) * 100, 2)
            elif ev_pct is not None:
                try:
                    ev_pct = float(ev_pct)
                except (TypeError, ValueError):
                    ev_pct = None
            rows.append({
                "Data": str(p.get("date") or "")[:10],
                "Ora": p.get("start_time_local") or "",
                "Torneo": p.get("tourney") or "",
                "Match": f"{p.get('player_a')} vs {p.get('player_b')}",
                "Superficie": p.get("surface") or "",
                "P(A)": p.get("p_win_a"),
                "Giocabilità": p.get("playability"),
                "Band": p.get("playability_label") or "",
                "Azione": p.get("action") or "no_bet",
                "Pick": rec.get("player") if rec else "",
                "Quota": rec.get("odds") if rec else None,
                # Numerico: evita ordinamento lessicografico ("+2%" > "+10%")
                "EV %": ev_pct,
                "KellyAdj": rec.get("kelly_adj_rank"),
                "Tour": p.get("tour") or "",
                "Fonte": p.get("odds_source") or "",
            })
        st.subheader("Calendario")
        df_cal = pd.DataFrame(rows).sort_values(
            ["Data", "Ora", "Giocabilità", "EV %", "Torneo"],
            ascending=[True, True, False, False, True],
            na_position="last",
        )
        st.dataframe(
            df_cal,
            use_container_width=True,
            hide_index=True,
            column_config={
                "EV %": st.column_config.NumberColumn(
                    "EV %",
                    help="Expected value in percentuale (ordinamento numerico)",
                    format="%+.1f%%",
                ),
                "Giocabilità": st.column_config.NumberColumn("Giocabilità", format="%.0f"),
                "P(A)": st.column_config.NumberColumn("P(A)", format="%.1%"),
                "Quota": st.column_config.NumberColumn("Quota", format="%.2f"),
                "KellyAdj": st.column_config.NumberColumn(
                    "KellyAdj",
                    help="Ranking Kelly × sostenibilità quota (non EV grezzo)",
                    format="%.4f",
                ),
            },
        )

        st.subheader("Dettaglio match")
        for p in sorted(preds, key=lambda x: float(x.get("playability") or 0), reverse=True)[:30]:
            rec = p.get("recommended")
            play = int(p.get("playability") or 0)
            icon = "✅" if play >= 75 and p.get("action") == "bet" else "⬜"
            with st.expander(
                f"{icon} [{play}/100] {p.get('player_a')} vs {p.get('player_b')} — {p.get('surface')}"
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("P(A)", f"{p.get('p_win_a', 0):.1%}")
                col2.metric("Markov", f"{p.get('p_markov', 0):.1%}")
                col3.metric("Elo", f"{p.get('p_elo', 0):.1%}")
                col4.metric("Giocabilità", f"{play}/100")
                if rec:
                    st.success(
                        f"**{rec['player']}** @ {rec['odds']} | "
                        f"EV {rec['ev']:+.1%} | Kelly {rec['kelly']:.2%} | "
                        f"Fonte: {p.get('odds_source', 'book')} | {p.get('playability_label', '')}"
                    )
                    parts = p.get("playability_parts") or {}
                    if parts:
                        st.caption(
                            "Componenti: value {value:.0%} · modelli {model_agreement:.0%} · "
                            "mercato {market_quality:.0%} · moneyway {moneyway:.0%} · drop {dropping_odds:.0%}".format(
                                **{k: parts.get(k, 0) for k in parts}
                            )
                        )
                else:
                    st.info("Nessun value bet sopra soglia")

with tab_back:
    cal_path = ROOT / "data" / "models" / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        bt = cal.get("backtest_summary", {})
        if bt.get("ok"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROI", f"{bt.get('roi', 0):.1%}")
            c2.metric("Hit Rate", f"{bt.get('hit_rate', 0):.1%}")
            c3.metric("N° Bet", bt.get("n_bets", 0))
            c4.metric("Max DD", f"{bt.get('max_drawdown', 0):.1%}")
            by_surf = bt.get("by_surface", [])
            if by_surf:
                st.subheader("Per superficie")
                st.dataframe(pd.DataFrame(by_surf), use_container_width=True)
        else:
            st.info(bt.get("error", "Backtest non ancora eseguito"))
    else:
        st.warning("Esegui `python main.py backtest` dopo il training")

with tab_data:
    st.subheader("Pipeline")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync base (Sackmann + Odds)"):
            import subprocess
            subprocess.run([sys.executable, "main.py", "sync", "--copy"], cwd=ROOT)
            st.success("Sync completato")
    with col2:
        if st.button("Sync extra (tutte le fonti)"):
            import subprocess
            subprocess.run([sys.executable, "main.py", "sync", "--copy", "--extra", "--force"], cwd=ROOT)
            st.success("Sync extra completato")

    st.subheader("Fonti dati integrate")
    sources = [
        ("CourtSpeed CPI", ROOT / "data" / "raw" / "courtspeed_cpi.json"),
        ("Tennis Abstract Elo", ROOT / "data" / "raw" / "tennis_abstract_elo.json"),
        ("TennisRatio skills", ROOT / "data" / "raw" / "tennisratio_rankings.json"),
        ("MCP Charting", ROOT / "data" / "raw" / "charting"),
        ("Wikidata players", ROOT / "data" / "processed" / "wikidata_players.json"),
        ("InfoTennis", ROOT / "data" / "raw" / "infotennis"),
        ("Seeder export", ROOT / "data" / "raw" / "seeder"),
    ]
    for name, path in sources:
        status = "OK" if path.exists() and (path.is_dir() and any(path.iterdir()) or path.is_file()) else "—"
        st.text(f"{name}: {status}")

    if st.button("Build + Features + Train"):
        with st.spinner("Pipeline in corso (può richiedere alcuni minuti)..."):
            import subprocess
            for cmd in ["build", "features", "train", "backtest"]:
                subprocess.run([sys.executable, "main.py", cmd], cwd=ROOT)
            st.success("Pipeline completata")
            st.rerun()

    matches_path = ROOT / "data" / "processed" / "matches.csv"
    if matches_path.exists():
        df = pd.read_csv(matches_path, nrows=5)
        st.subheader("Anteprima matches.csv")
        st.dataframe(df, use_container_width=True)
        st.caption(f"Totale righe: {sum(1 for _ in open(matches_path)) - 1}")

with tab_elo:
    st.subheader("Elo Tennis Abstract (ATP + WTA)")
    ta_atp = ROOT / "data" / "raw" / "tennis_abstract_elo.json"
    ta_wta = ROOT / "data" / "raw" / "tennis_abstract_elo_wta.json"
    if st.button("Scarica Elo da Tennis Abstract"):
        from modules.data_update.tennis_abstract import fetch_all_tennis_abstract_elo
        result = fetch_all_tennis_abstract_elo(force=True)
        st.json(result)
    for label, path in (("ATP", ta_atp), ("WTA", ta_wta)):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            players = data.get("players", [])[:30]
            if players:
                st.caption(f"{label} — top 30")
                st.dataframe(pd.DataFrame(players), use_container_width=True)
