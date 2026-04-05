import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import uuid
import time
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.style import inject_css

try:
    import quant_reporter as qr
    QUANT_REPORTER_AVAILABLE = True
except ImportError:
    QUANT_REPORTER_AVAILABLE = False

API_URL = "http://127.0.0.1:8000/api"
COLORS  = ["#00D4FF", "#00FF94", "#FFB800", "#FF4466", "#BF7FFF", "#FF8C00"]
SPEED_MAP = {"Fast": 5000, "Medium": 20000, "Thorough": 50000}

st.set_page_config(page_title="Portfolio — QuantVis", page_icon="▲", layout="wide")
inject_css()

st.markdown("""
<div style="border-bottom:1px solid #1E2D3D;padding-bottom:1rem;margin-bottom:1.5rem;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:0.2em;color:#00D4FF;text-transform:uppercase;">PORTFOLIO MANAGEMENT</span>
    <h1 style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:500;color:#E8F4FD;margin:0.3rem 0 0 0;">
        Portfolio Manager
    </h1>
</div>
""", unsafe_allow_html=True)

tab_optimizer, tab_report = st.tabs([
    "ALLOCATOR",
    "DEEP REPORT  (BACKTEST)",
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _blend(quant: dict, rl: dict, rl_w: float) -> dict:
    all_t = set(quant) | set(rl)
    b = {t: (1 - rl_w) * quant.get(t, 0.0) + rl_w * rl.get(t, 0.0) for t in all_t}
    total = sum(b.values())
    if total > 0:
        b = {t: round(v / total, 4) for t, v in b.items()}
    return {t: v for t, v in sorted(b.items(), key=lambda x: -x[1]) if v > 0.001}


def _donut(alloc: dict, centre: str):
    fig = go.Figure(data=[go.Pie(
        labels=list(alloc.keys()), values=list(alloc.values()), hole=0.6,
        marker=dict(colors=COLORS[:len(alloc)], line=dict(color="#080C10", width=2)),
        textfont=dict(family="IBM Plex Mono", size=10, color="#E8F4FD"),
    )])
    fig.update_layout(
        paper_bgcolor="#080C10", plot_bgcolor="#080C10",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(font=dict(family="IBM Plex Mono", size=10, color="#7A9BB5"),
                    bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=centre, x=0.5, y=0.5, font_size=13, showarrow=False,
                          font=dict(family="IBM Plex Mono", color="#00D4FF"))],
    )
    return fig


def _poll_rl(session_id, progress_bar, status_text, max_polls=400):
    LABELS = {
        "fetching_data":         "Fetching 2 years of price history...",
        "fetching_fundamentals": "Scoring fundamentals & news sentiment via AI...",
        "training_A2C":          "[1/3] Training A2C — best cumulative returns...",
        "training_SAC":          "[2/3] Training SAC — entropy-driven diversification...",
        "training_TD3":          "[3/3] Training TD3 — balanced, bias-corrected...",
        "evaluating":            "Computing ensemble allocation...",
    }
    for _ in range(max_polls):
        try:
            r = requests.get(f"{API_URL}/rl/progress/{session_id}/", timeout=5)
            if r.status_code == 200:
                p      = r.json()
                status = p.get("status", "")
                pct    = p.get("progress", 0)
                progress_bar.progress(int(pct))
                status_text.markdown(
                    f'<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#00FF94;">'
                    f'▸ {LABELS.get(status, status)}  {pct:.0f}%</div>',
                    unsafe_allow_html=True,
                )
                if p.get("status") == "error":
                    st.error(p.get("error", "Unknown error"))
                    return None
                if p.get("status") == "complete":
                    return p
        except Exception:
            pass
        time.sleep(2)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — ALLOCATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_optimizer:

    # ── How it works explanation ──────────────────────────────────────────────
    with st.expander("HOW THIS ALLOCATOR WORKS", expanded=False):
        st.markdown("""
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.85rem;color:#7A9BB5;line-height:1.8;">

        <b style="color:#00D4FF;">What it does</b><br>
        This allocator combines two completely different approaches to portfolio construction
        and lets you blend them into a single final allocation.

        <br><br>

        <b style="color:#00D4FF;">Step 1 — Quant Optimisation (Riskfolio-Lib)</b><br>
        Uses 2 years of historical price data to mathematically optimise weights based on
        your chosen risk strategy:
        <ul style="margin-top:0.3rem;">
            <li><b style="color:#E8F4FD;">Conservative (HRP)</b> — Hierarchical Risk Parity. Clusters assets by correlation and allocates inversely to risk. No matrix inversion, robust to estimation error.</li>
            <li><b style="color:#E8F4FD;">Balanced (Min CVaR)</b> — Minimises Conditional Value at Risk (the average loss in the worst 5% of scenarios). Best for tail-risk protection.</li>
            <li><b style="color:#E8F4FD;">Aggressive (Max Sharpe)</b> — Maximises return per unit of risk. Concentrates in historically high-performing assets.</li>
        </ul>

        <b style="color:#00D4FF;">Step 2 — RL Ensemble (FinRL, NeurIPS 2020)</b><br>
        Trains 3 reinforcement learning agents simultaneously on the same historical data.
        Each agent learns to rebalance daily to maximise a risk-adjusted reward:
        <ul style="margin-top:0.3rem;">
            <li><b style="color:#E8F4FD;">A2C</b> — On-policy agent. Best for capturing overall cumulative returns across market regimes.</li>
            <li><b style="color:#E8F4FD;">SAC</b> — Off-policy with built-in entropy maximisation. Naturally avoids concentration and promotes diversification.</li>
            <li><b style="color:#E8F4FD;">TD3</b> — Twin-critic off-policy agent. Eliminates overestimation bias, produces the most balanced and stable allocations.</li>
        </ul>
        The final RL allocation is a weighted average: 35% A2C + 30% SAC + 35% TD3.<br><br>

        Each agent's reward function includes:
        <ul style="margin-top:0.3rem;">
            <li>Daily portfolio return minus 0.1% transaction cost per rebalance</li>
            <li>Sortino ratio bonus — rewards consistency, ignores upside volatility</li>
            <li>Sector-risk-adjusted drawdown penalty — Tech/Energy penalised more than Utilities/Healthcare</li>
            <li>Entropy bonus — discourages concentration</li>
            <li>Fundamental alignment — overweighting strong PE/debt/growth stocks is rewarded</li>
            <li>News sentiment alignment — overweighting positively-covered stocks is rewarded</li>
            <li>Hard 45% cap per asset enforced on every step</li>
        </ul>

        <b style="color:#00D4FF;">Step 3 — Blend</b><br>
        The slider controls how much of each method feeds into the final allocation:<br>
        <code style="color:#FFB800;">Final weight = (1 − blend) × Quant + blend × RL Ensemble</code><br>
        At 0% RL you get pure Riskfolio. At 100% RL you get pure ensemble. In between you get both.

        </div>
        """, unsafe_allow_html=True)

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_tickers, col_controls = st.columns([3, 2])
    with col_tickers:
        tickers_input = st.text_area("TICKERS", "AAPL, GOOG, MSFT, TSLA", height=80)
    with col_controls:
        risk_choice = st.select_slider(
            "RISK STRATEGY",
            options=["Conservative (HRP)", "Balanced (Min CVaR)", "Aggressive (Max Sharpe)"],
            value="Balanced (Min CVaR)",
        )
        RISK_MAP = {
            "Conservative (HRP)":       0.2,
            "Balanced (Min CVaR)":      0.5,
            "Aggressive (Max Sharpe)":  0.8,
        }
        risk_val       = RISK_MAP[risk_choice]
        strategy_short = risk_choice.split("(")[1].replace(")", "").strip()

    # ── Blend + speed ─────────────────────────────────────────────────────────
    col_blend, col_speed = st.columns([3, 1])
    with col_blend:
        st.markdown('<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#7A9BB5;text-transform:uppercase;margin-bottom:0.2rem;">QUANT ← BLEND → RL ENSEMBLE</div>', unsafe_allow_html=True)
        rl_blend   = st.slider("blend", 0.0, 1.0, 0.5, step=0.05, label_visibility="collapsed")
        quant_pct  = round((1 - rl_blend) * 100)
        rl_pct     = round(rl_blend * 100)
        st.markdown(
            f'<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#7A9BB5;margin-top:-0.3rem;">'
            f'<span style="color:#00D4FF;">{quant_pct}% {strategy_short}</span>'
            f'  +  '
            f'<span style="color:#00FF94;">{rl_pct}% RL Ensemble (A2C · SAC · TD3)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_speed:
        speed        = st.selectbox("RL TRAINING", ["Fast", "Medium", "Thorough"], index=0)
        rl_timesteps = SPEED_MAP[speed]

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  OPTIMISE & BLEND", type="primary")

    # Session state
    for k, v in [
        ("opt_quant", None), ("opt_rl",    None),
        ("opt_blend", None), ("opt_reason",None),
        ("opt_feats", None), ("opt_sid",   str(uuid.uuid4())[:8]),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    if run_btn:
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.warning("Enter at least 2 tickers.")
        else:
            st.session_state.opt_quant  = None
            st.session_state.opt_rl     = None
            st.session_state.opt_blend  = None
            st.session_state.opt_sid    = str(uuid.uuid4())[:8]

            # Step 1 — Riskfolio
            with st.spinner(f"Step 1 — Running {strategy_short} optimisation..."):
                try:
                    resp = requests.post(f"{API_URL}/optimize/",
                                         json={"tickers": tickers, "risk_tolerance": risk_val})
                    if resp.status_code == 200:
                        d = resp.json()
                        st.session_state.opt_quant  = d.get("allocation", {})
                        st.session_state.opt_reason = d.get("reasoning", "")
                    else:
                        st.error("Riskfolio optimisation failed.")
                except Exception as e:
                    st.error(f"Connection error: {e}")

            # Step 2 — RL Ensemble
            if st.session_state.opt_quant and rl_blend > 0:
                st.markdown(
                    '<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#00FF94;'
                    'text-transform:uppercase;margin:1rem 0 0.5rem 0;">'
                    'Step 2 — Training RL Ensemble (A2C · SAC · TD3)</div>',
                    unsafe_allow_html=True,
                )
                pb   = st.progress(0)
                stxt = st.empty()
                try:
                    kick = requests.post(f"{API_URL}/rl/train/", json={
                        "tickers":    tickers,
                        "timesteps":  rl_timesteps,
                        "session_id": st.session_state.opt_sid,
                    })
                    if kick.status_code == 200:
                        rl_res = _poll_rl(st.session_state.opt_sid, pb, stxt)
                        if rl_res:
                            st.session_state.opt_rl    = rl_res.get("allocation", {})
                            st.session_state.opt_feats = rl_res.get("ticker_features", {})
                    else:
                        st.warning("RL failed — showing pure Quant result.")
                except Exception as e:
                    st.warning(f"RL error: {e}")

            # Blend
            q = st.session_state.opt_quant or {}
            r = st.session_state.opt_rl    or {}
            if q or r:
                st.session_state.opt_blend = _blend(q, r, rl_blend) if (q and r) else (q or r)
            st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.opt_blend:
        q      = st.session_state.opt_quant or {}
        r      = st.session_state.opt_rl    or {}
        blend  = st.session_state.opt_blend
        feats  = st.session_state.opt_feats or {}
        reason = st.session_state.opt_reason or ""

        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
        color:#00FF94;text-transform:uppercase;margin:1.5rem 0 1rem 0;">▸ BLENDED ALLOCATION READY</div>
        """, unsafe_allow_html=True)

        # Three donuts
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:0.6rem;color:#00D4FF;text-align:center;margin-bottom:0.3rem;">{strategy_short} — {quant_pct}%</div>', unsafe_allow_html=True)
            if q:
                st.plotly_chart(_donut(q, strategy_short[:4]), use_container_width=True)
        with c2:
            st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:0.6rem;color:#00FF94;text-align:center;margin-bottom:0.3rem;">RL Ensemble — {rl_pct}%</div>', unsafe_allow_html=True)
            if r:
                st.plotly_chart(_donut(r, "ENS"), use_container_width=True)
            else:
                st.caption("Blend set to 0% — RL not trained")
        with c3:
            st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:0.6rem;color:#FFB800;text-align:center;margin-bottom:0.3rem;">Final Blend — {quant_pct}% + {rl_pct}%</div>', unsafe_allow_html=True)
            st.plotly_chart(_donut(blend, "MIX"), use_container_width=True)

        # Comparison table
        st.markdown("<br>", unsafe_allow_html=True)
        all_t = sorted(set(list(q.keys()) + list(r.keys()) + list(blend.keys())))
        rows  = []
        for t in all_t:
            qw = q.get(t, 0.0)
            rw = r.get(t, 0.0)
            bw = blend.get(t, 0.0)
            delta = bw - qw
            rows.append({
                "Ticker":                       t,
                f"Quant ({strategy_short})":    f"{qw:.1%}" if qw > 0 else "—",
                "RL Ensemble":                  f"{rw:.1%}" if rw > 0 else "—",
                "Final Blend":                  f"{bw:.1%}" if bw > 0 else "—",
                "Δ RL vs Quant":                (f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}") if qw > 0 else "new",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Signal scorecard
        if feats:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#7A9BB5;text-transform:uppercase;margin-bottom:0.5rem;">▸ RL SIGNAL SCORECARD — WHAT DROVE THE RL ALLOCATION</div>', unsafe_allow_html=True)
            srows = []
            for ticker, f in feats.items():
                fs = f.get("fundamental_score", 0.5)
                sv = f.get("news_sentiment",    0.0)
                sr = f.get("sector_risk",       1.0)
                srows.append({
                    "Ticker":      ticker,
                    "Final Weight":f"{blend.get(ticker, 0):.1%}",
                    "Fundamentals":"🟢 Strong"  if fs > 0.65 else ("🟡 OK"      if fs > 0.4  else "🔴 Weak"),
                    "News":        "🟢 Bullish" if sv > 0.3  else ("🔴 Bearish" if sv < -0.3  else "🟡 Neutral"),
                    "Sector Risk": "🔴 High"    if sr >= 1.3 else ("🟡 Medium"  if sr >= 1.0  else "🟢 Low"),
                    "Fund Score":  f"{fs:.2f}",
                    "Sentiment":   f"{sv:+.2f}",
                    "Risk ×":      f"{sr:.1f}×",
                })
            st.dataframe(
                pd.DataFrame(srows).sort_values("Final Weight", ascending=False),
                use_container_width=True, hide_index=True,
            )

        # AI commentary
        if reason:
            st.markdown(f"""
            <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
            border-left:3px solid #00D4FF;padding:1rem;margin-top:1rem;
            font-family:'IBM Plex Sans',sans-serif;font-size:0.82rem;color:#7A9BB5;line-height:1.7;">
                <span style="font-family:IBM Plex Mono;font-size:0.6rem;color:#00D4FF;text-transform:uppercase;">
                ▸ AI COMMENTARY ({strategy_short})</span><br><br>{reason}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(255,184,0,0.05);border:1px solid rgba(255,184,0,0.15);
        border-left:3px solid #FFB800;padding:0.8rem 1rem;margin-top:1rem;">
            <span style="font-family:IBM Plex Mono;font-size:0.65rem;color:#FFB800;">▸ NOTE</span>
            <span style="font-family:IBM Plex Sans;font-size:0.8rem;color:#7A9BB5;margin-left:0.5rem;">
                Past performance does not guarantee future results. RL allocation is based on historical patterns only.
                Use alongside your own fundamental research.
            </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DEEP REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_report:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;color:#7A9BB5;text-transform:uppercase;margin-bottom:1rem;">PROFESSIONAL QUANT REPORT — BACKTESTING · EFFICIENT FRONTIER · MONTE CARLO</div>""", unsafe_allow_html=True)
    if not QUANT_REPORTER_AVAILABLE:
        st.error("`quant-reporter` not installed.")
    else:
        with st.expander("REPORT CONFIGURATION", expanded=True):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                repo_tickers = st.text_input("Portfolio Tickers", "AAPL, MSFT, SPY, QQQ")
                benchmark    = st.text_input("Benchmark Ticker",  "SPY")
            with col_in2:
                default_start = datetime.now() - timedelta(days=365 * 2)
                start_date    = st.date_input("Start Date", default_start)
                end_date      = st.date_input("End Date",   datetime.now() - timedelta(days=90))
        if st.button("▶  GENERATE DEEP REPORT", type="primary"):
            r_tickers = [t.strip() for t in repo_tickers.split(",") if t.strip()]
            if not r_tickers:
                st.warning("Enter tickers.")
            else:
                with st.spinner("Running analysis... (30–60s)"):
                    try:
                        import os as _os
                        report_file = _os.path.abspath("portfolio_report.html")
                        qr.create_combined_report(
                            portfolio_dict={t: 1.0/len(r_tickers) for t in r_tickers},
                            benchmark_ticker=benchmark,
                            train_start=start_date.strftime("%Y-%m-%d"),
                            train_end=end_date.strftime("%Y-%m-%d"),
                            filename=report_file,
                        )
                        with open(report_file, "r", encoding="utf-8") as f:
                            html_content = f.read()
                        st.success("Report generated.")
                        st.download_button("DOWNLOAD REPORT", html_content,
                                           "Quant_Report.html", "text/html")
                        st.divider()
                        components.html(html_content, height=800, scrolling=True)
                    except Exception as e:
                        st.error(f"Failed: {e}")
