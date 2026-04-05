import streamlit as st
import requests as http_requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import concurrent.futures
import streamlit.components.v1 as components

# ── backend path ──────────────────────────────────────────────────────
_backend = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
if os.path.isdir(_backend) and _backend not in sys.path:
    sys.path.insert(0, _backend)

API_URL = "http://127.0.0.1:8000/api"

# ── quant_reporter (optional) ─────────────────────────────────────────
try:
    import quant_reporter as qr
    import yfinance as yf

    if hasattr(qr, 'opt_core'):
        def patched_get_risk_free_rate():
            try:
                tbill = yf.download("^IRX", period="5d")
                if tbill is None or tbill.empty:
                    raise Exception("^IRX download failed")
                latest_rate = float(np.ravel(tbill['Close'].iloc[-1])[0]) / 100
                if not 0 <= latest_rate <= 0.2:
                    raise Exception("Rate unrealistic.")
                return latest_rate
            except Exception:
                return 0.06
        qr.opt_core.get_risk_free_rate = patched_get_risk_free_rate

        def patched_calculate_rolling_returns(cumulative_df):
            periods = {'1-Year': 252, '3-Year': 252*3, '5-Year': 252*5}
            rolling_returns = {}
            for name, days in periods.items():
                if len(cumulative_df) > days:
                    years = days / 252
                    end_val = float(np.ravel(cumulative_df.iloc[-1])[0])
                    start_val = float(np.ravel(cumulative_df.iloc[-days-1])[0])
                    rolling_returns[name] = [(end_val / start_val)**(1/years) - 1]
                else:
                    rolling_returns[name] = [np.nan]
            return (pd.DataFrame.from_dict(rolling_returns, orient='index')
                    .map(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A"))
        qr.opt_core.calculate_rolling_returns = patched_calculate_rolling_returns

    QUANT_REPORTER_AVAILABLE = True
except ImportError:
    QUANT_REPORTER_AVAILABLE = False

# ── page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio — QuantVision", layout="wide")

# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg,#0d1117,#161b22);
    border: 1px solid #21262d; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}
div[data-testid="stMetric"] label { color:#8b949e!important; font-size:0.75rem!important; text-transform:uppercase; letter-spacing:0.5px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#e6edf3!important; font-weight:700!important; }

/* SWOT quadrant cards */
.swot-card { border-radius:16px; padding:22px 24px; min-height:240px;
             box-shadow:0 6px 30px rgba(0,0,0,0.35); transition:transform 0.2s; }
.swot-card:hover { transform:translateY(-2px); }
.swot-S { background:linear-gradient(145deg,#0a1f0f,#0d2f18); border:1px solid #2a6b3a; }
.swot-W { background:linear-gradient(145deg,#1f1505,#2d1e07); border:1px solid #7a5c1e; }
.swot-O { background:linear-gradient(145deg,#040f1f,#081a35); border:1px solid #1a4a8a; }
.swot-T { background:linear-gradient(145deg,#1f0507,#350a0d); border:1px solid #8a1a1f; }
.swot-header { font-size:1.05rem; font-weight:800; margin-bottom:12px; }
.swot-S .swot-header { color:#4ade80; }
.swot-W .swot-header { color:#fbbf24; }
.swot-O .swot-header { color:#60a5fa; }
.swot-T .swot-header { color:#f87171; }
.swot-point { padding:9px 11px; border-radius:8px; margin-bottom:8px; font-size:0.83rem; line-height:1.5; }
.swot-S .swot-point { background:rgba(74,222,128,0.07); border-left:3px solid #4ade80; color:#dcfce7; }
.swot-W .swot-point { background:rgba(251,191,36,0.07); border-left:3px solid #fbbf24; color:#fef3c7; }
.swot-O .swot-point { background:rgba(96,165,250,0.07); border-left:3px solid #60a5fa; color:#dbeafe; }
.swot-T .swot-point { background:rgba(248,113,113,0.07); border-left:3px solid #f87171; color:#fee2e2; }
.swot-evidence { font-size:0.72rem; color:#94a3b8; margin-top:3px; font-style:italic; }
.swot-source   { font-size:0.68rem; color:#64748b; margin-top:2px; }

/* rating badges */
.rating-sb { background:#052e16;color:#4ade80;padding:7px 20px;border-radius:9px;font-weight:800;font-size:1rem;border:1px solid #4ade80;display:inline-block; }
.rating-b  { background:#0a2215;color:#86efac;padding:7px 20px;border-radius:9px;font-weight:800;font-size:1rem;border:1px solid #86efac;display:inline-block; }
.rating-h  { background:#1c1407;color:#fbbf24;padding:7px 20px;border-radius:9px;font-weight:800;font-size:1rem;border:1px solid #fbbf24;display:inline-block; }
.rating-s  { background:#2d0a0c;color:#f87171;padding:7px 20px;border-radius:9px;font-weight:800;font-size:1rem;border:1px solid #f87171;display:inline-block; }
.rating-ss { background:#1a0507;color:#ef4444;padding:7px 20px;border-radius:9px;font-weight:800;font-size:1rem;border:1px solid #ef4444;display:inline-block; }

/* flag chips */
.flag-green  { background:#052e16;color:#4ade80;padding:1px 7px;border-radius:4px;font-size:0.68rem;font-weight:700;display:inline-block;margin-left:5px; }
.flag-orange { background:#1c1407;color:#fbbf24;padding:1px 7px;border-radius:4px;font-size:0.68rem;font-weight:700;display:inline-block;margin-left:5px; }
.flag-blue   { background:#040f1f;color:#60a5fa;padding:1px 7px;border-radius:4px;font-size:0.68rem;font-weight:700;display:inline-block;margin-left:5px; }
.flag-red    { background:#2d0a0c;color:#f87171;padding:1px 7px;border-radius:4px;font-size:0.68rem;font-weight:700;display:inline-block;margin-left:5px; }

/* news card */
.news-card { background:#0d1117; border:1px solid #21262d; border-radius:9px;
             padding:12px 14px; margin-bottom:8px; }
.news-title { font-size:0.87rem; font-weight:600; color:#e6edf3; }
.news-meta  { font-size:0.72rem; color:#8b949e; margin-top:3px; }

/* section title */
.sec-title { font-size:0.95rem; font-weight:700; color:#e6edf3;
             border-left:3px solid #388bfd; padding-left:9px; margin:18px 0 10px 0; }

/* exec box */
.exec-box { background:linear-gradient(135deg,#040d1a,#071527);
            border:1px solid #1a3a5c; border-radius:12px; padding:16px 20px; margin:10px 0; }
.exec-text { color:#93c5fd; font-size:0.93rem; line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers (SWOT rendering, same logic as 7_SWOT_Analysis.py)
# ═══════════════════════════════════════════════════════════════════════

def rating_badge_html(rating: str) -> str:
    r = rating.upper().strip()
    cls_map = {"STRONG BUY": "rating-sb", "BUY": "rating-b",
               "HOLD": "rating-h", "SELL": "rating-s", "STRONG SELL": "rating-ss"}
    icon_map = {"STRONG BUY": "⬆⬆", "BUY": "⬆", "HOLD": "◆", "SELL": "⬇", "STRONG SELL": "⬇⬇"}
    for key in cls_map:
        if key in r:
            return f'<span class="{cls_map[key]}">{icon_map[key]} {key}</span>'
    return f'<span class="rating-h">◆ {rating}</span>'


def flag_chip(flag: str) -> str:
    mapping = {
        "GREEN":  ("flag-green",  "✓ Strength"),
        "ORANGE": ("flag-orange", "⚠ Caution"),
        "BLUE":   ("flag-blue",   "✦ Opportunity"),
        "RED":    ("flag-red",    "⚡ Risk"),
    }
    cls, label = mapping.get(flag.upper(), ("flag-blue", flag))
    return f'<span class="{cls}">{label}</span>'


def render_swot_card(quadrant_key: str, title: str, emoji: str, points: list):
    css = {"S": "swot-S", "W": "swot-W", "O": "swot-O", "T": "swot-T"}[quadrant_key]
    pts_html = ""
    for p in points:
        chip = flag_chip(p.get("flag","")) if p.get("flag") else ""
        pts_html += f"""<div class="swot-point">
<strong>{p.get("point","")}</strong>{chip}
{f'<div class="swot-evidence">📊 {p["evidence"]}</div>' if p.get("evidence") else ""}
{f'<div class="swot-source">📎 {p["source"]}</div>'    if p.get("source")   else ""}
</div>"""
    st.markdown(f"""<div class="swot-card {css}">
<div class="swot-header">{emoji} {title}</div>{pts_html}
</div>""", unsafe_allow_html=True)


def call_swot_api(query: str) -> dict:
    """Call the SWOT backend — HTTP first, direct import fallback."""
    try:
        resp = http_requests.post(f"{API_URL}/swot/",
                                  json={"query": query}, timeout=90)
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.json().get("error", "API error")}
    except http_requests.exceptions.ConnectionError:
        try:
            from api.swot_service import run_swot_analysis
            return run_swot_analysis(query)
        except Exception as e:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def render_full_swot_block(data: dict):
    """
    Render the complete SWOT output (same layout as 7_SWOT_Analysis.py)
    for embedding inside the Portfolio Deep Report tab.
    """
    if not data or "error" in data:
        st.error(f"SWOT Error: {data.get('error','Unknown')}")
        return

    fund  = data.get("fundamentals", {})
    pm    = data.get("price_metrics", {})
    news  = data.get("news", [])
    peers = data.get("peers", [])
    swot  = data.get("swot", {})
    curr  = data.get("currency", "$")
    cname = data.get("company_name", data.get("ticker", ""))
    mkt   = data.get("market", "US")
    gen   = data.get("generated_at", "")
    ticker_sym = data.get("ticker", "")

    # ── Company header ────────────────────────────────────────────────
    mkt_flag = "🇮🇳 NSE/BSE · ₹" if mkt == "IN" else "🇺🇸 NYSE/NASDAQ · $"
    price_val = fund.get("price")
    day_chg   = fund.get("day_change_pct")

    h1, h2, h3 = st.columns([3, 1.5, 1.5])
    with h1:
        st.markdown(f"""<div style="padding:14px 18px;background:#0d1117;border:1px solid #21262d;border-radius:11px;">
<div style="font-size:1.35rem;font-weight:800;color:#e6edf3;">{cname}</div>
<div style="color:#8b949e;font-size:0.83rem;margin-top:4px;">
{ticker_sym} · {mkt_flag} · {fund.get("sector","N/A")} · {fund.get("industry","N/A")}
</div>
<div style="color:#484f58;font-size:0.72rem;margin-top:5px;">Generated: {gen}</div>
</div>""", unsafe_allow_html=True)
    with h2:
        if price_val:
            st.metric("Current Price", f"{curr}{price_val:,.2f}",
                      f"{day_chg:+.2f}%" if day_chg else None)
    with h3:
        target = fund.get("analyst_target")
        if target:
            st.metric("Analyst Target", f"{curr}{target:,.2f}")
        st.markdown(f"**Consensus:** `{(fund.get('recommendation') or 'N/A').upper()}`")

    # ── Investment rating + summary ───────────────────────────────────
    if swot.get("investment_rating"):
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1.8, 3, 2])
        with r1:
            st.markdown('<div class="sec-title">Investment Rating</div>', unsafe_allow_html=True)
            st.markdown(rating_badge_html(swot["investment_rating"]), unsafe_allow_html=True)
        with r2:
            if swot.get("executive_summary"):
                st.markdown(f'<div class="exec-box"><div class="exec-text">'
                            f'{swot["executive_summary"]}</div></div>', unsafe_allow_html=True)
        with r3:
            if swot.get("key_risk"):
                st.error(f"⚡ {swot['key_risk']}")
            if swot.get("key_opportunity"):
                st.success(f"✦ {swot['key_opportunity']}")

    # ── Financial snapshot ────────────────────────────────────────────
    st.markdown('<div class="sec-title">Financial Snapshot</div>', unsafe_allow_html=True)
    snap_metrics = [
        ("Market Cap",     fund.get("market_cap_fmt","N/A")),
        ("P/E Ratio",      str(round(fund["pe_ratio"],1)) if fund.get("pe_ratio") else "N/A"),
        ("ROE",            f"{fund['roe']*100:.1f}%"      if fund.get("roe") else "N/A"),
        ("Revenue",        fund.get("revenue_fmt","N/A")),
        ("Net Income",     fund.get("net_income_fmt","N/A")),
        ("Profit Margin",  f"{fund['profit_margin']*100:.1f}%" if fund.get("profit_margin") else "N/A"),
        ("Rev. Growth",    f"{fund['revenue_growth']*100:.1f}%" if fund.get("revenue_growth") else "N/A"),
        ("Debt/Equity",    str(round(fund["debt_equity"],1)) if fund.get("debt_equity") else "N/A"),
        ("Free Cash Flow", fund.get("free_cashflow_fmt","N/A")),
        ("Beta",           str(round(fund["beta"],2))     if fund.get("beta") else "N/A"),
        ("52W High",       f"{curr}{fund['week52_high']:,.2f}" if fund.get("week52_high") else "N/A"),
        ("52W Low",        f"{curr}{fund['week52_low']:,.2f}"  if fund.get("week52_low") else "N/A"),
    ]
    if pm:
        snap_metrics += [
            ("52W Perf.", f"{pm.get('perf_52w','N/A')}%"),
            ("Ann. Vol.",  f"{pm.get('ann_volatility_pct','N/A')}%"),
            ("RSI (14)",   str(pm.get("rsi_14","N/A"))),
            ("Analyst Rec", (fund.get("recommendation") or "N/A").upper()),
        ]
    snap_cols = st.columns(4)
    for i, (label, val) in enumerate(snap_metrics):
        with snap_cols[i % 4]:
            st.metric(label, val)

    # ── Price chart ───────────────────────────────────────────────────
    if pm and pm.get("close_prices") and pm.get("dates"):
        prices = pm["close_prices"]
        dates  = pm["dates"]
        color  = "#4ade80" if prices[-1] >= prices[0] else "#f87171"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices, fill="tozeroy",
            fillcolor="rgba(74,222,128,0.07)" if color == "#4ade80" else "rgba(248,113,113,0.07)",
            line=dict(color=color, width=2),
            hovertemplate=f"{curr}%{{y:,.2f}}<extra></extra>",
        ))
        pct = (prices[-1]/prices[0]-1)*100 if prices[0] else 0
        fig.update_layout(
            title=dict(text=f"{cname} — 60-Day Price ({'+' if pct>=0 else ''}{pct:.2f}%)",
                       font=dict(color="#e6edf3", size=13)),
            height=240, template="plotly_dark",
            plot_bgcolor="rgba(13,17,23,0.9)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40,r=20,t=40,b=30),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#21262d"),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── SWOT quadrants ────────────────────────────────────────────────
    st.markdown("""
    <h3 style="background:linear-gradient(90deg,#4ade80,#60a5fa,#f59e0b,#f87171);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-weight:900;font-size:1.4rem;margin:16px 0 12px 0;">SWOT Analysis</h3>
    """, unsafe_allow_html=True)

    if "error" in swot:
        st.error(f"SWOT generation error: {swot['error']}")
    else:
        sw1, sw2 = st.columns(2, gap="large")
        with sw1:
            render_swot_card("S", "STRENGTHS",    "💪", swot.get("strengths",    []))
            st.markdown("<br>", unsafe_allow_html=True)
            render_swot_card("O", "OPPORTUNITIES","🚀", swot.get("opportunities",[]))
        with sw2:
            render_swot_card("W", "WEAKNESSES",   "⚠️", swot.get("weaknesses",   []))
            st.markdown("<br>", unsafe_allow_html=True)
            render_swot_card("T", "THREATS",      "⚡", swot.get("threats",      []))

    # ── Peer comparison ───────────────────────────────────────────────
    if peers:
        st.markdown('<div class="sec-title">Competitor Benchmarking</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(peers), use_container_width=True, hide_index=True)

    # ── Recent news ───────────────────────────────────────────────────
    if news:
        st.markdown(f'<div class="sec-title">Latest News ({len(news)} items)</div>',
                    unsafe_allow_html=True)
        for n in news[:8]:
            st.markdown(f"""<div class="news-card">
<div class="news-title">
<a href="{n.get('link','#')}" target="_blank"
style="color:#e6edf3;text-decoration:none;">{n.get('title','')}</a>
</div>
<div class="news-meta">📰 {n.get('source','Unknown')} · 🕐 {n.get('date','')}</div>
</div>""", unsafe_allow_html=True)

    st.caption("⚠️ SWOT generated by Groq AI from live market data. Not financial advice.")


def build_swot_html_for_report(ticker: str, data: dict) -> str:
    if not data or "error" in data:
        return f'<div style="margin-bottom: 30px; font-family: sans-serif; padding: 20px; border: 1px solid #333; border-radius: 8px;"><h3>{ticker}</h3><p style="color: #f87171;">Error fetching SWOT: {data.get("error", "Unknown error")}</p></div>'

    swot = data.get("swot", {})
    if "error" in swot:
        return f'<div style="margin-bottom: 30px; font-family: sans-serif; padding: 20px; border: 1px solid #333; border-radius: 8px;"><h3>{ticker}</h3><p style="color: #f87171;">Generation Error: {swot["error"]}</p></div>'

    cname = data.get("company_name", ticker)
    price = data.get("fundamentals", {}).get("price", "N/A")
    curr = data.get("currency", "$")
    
    def gen_card(title, emoji, points, border_col, bg_col, text_col):
        pts_html = ""
        for p in points:
            pts_html += f'<div style="padding: 8px; margin-bottom: 6px; background: rgba(255,255,255,0.05); border-left: 3px solid {border_col}; border-radius: 6px; font-size: 0.9rem; color: #ddd;"><strong>{p.get("point","")}</strong><br><span style="font-size:0.75rem; color: #999;">📊 {p.get("evidence","")}</span></div>'
        return f'<div style="flex: 1; min-width: 300px; padding: 20px; border-radius: 12px; border: 1px solid {border_col}; background: {bg_col}; margin: 10px;"><h3 style="color: {text_col}; margin-top: 0;">{emoji} {title}</h3>{pts_html}</div>'
    
    s_card = gen_card("STRENGTHS", "💪", swot.get("strengths", []), "#2a6b3a", "#0d2f18", "#4ade80")
    w_card = gen_card("WEAKNESSES", "⚠️", swot.get("weaknesses", []), "#7a5c1e", "#2d1e07", "#fbbf24")
    o_card = gen_card("OPPORTUNITIES", "🚀", swot.get("opportunities", []), "#1a4a8a", "#081a35", "#60a5fa")
    t_card = gen_card("THREATS", "⚡", swot.get("threats", []), "#8a1a1f", "#350a0d", "#f87171")

    return f'''
    <div style="font-family: sans-serif; color: #eee; background: #0d1117; padding: 30px; border-radius: 12px; margin-bottom: 40px; border: 1px solid #30363d;">
        <h2 style="margin-top: 0; border-bottom: 1px solid #30363d; padding-bottom: 10px; color: #e6edf3;">{cname} ({ticker}) <span style="font-size: 1rem; color: #8b949e; margin-left: 10px;">Current Price: {curr}{price}</span></h2>
        <div style="margin-bottom: 20px; padding: 15px; background: #161b22; border-radius: 8px; border: 1px solid #30363d;">
            <h4 style="margin: 0 0 10px 0; color: #c9d1d9;">Executive Summary</h4>
            <p style="margin: 0; line-height: 1.6; color: #8b949e;">{swot.get("executive_summary", "")}</p>
        </div>
        <div style="display: flex; flex-wrap: wrap;">
            {s_card}
            {w_card}
        </div>
        <div style="display: flex; flex-wrap: wrap;">
            {o_card}
            {t_card}
        </div>
    </div>
    '''


# ═══════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style="background:linear-gradient(90deg,#58a6ff,#bc8cff,#3fb950);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-weight:900;font-size:2rem;margin:0 0 4px 0;">
    Portfolio Manager
</h1>
<p style="color:#8b949e;font-size:0.95rem;margin:0;">
    Optimizer · Deep Quant Report · AI SWOT Intelligence
</p>
""", unsafe_allow_html=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab_optimizer, tab_report = st.tabs([
    "⚖️ Allocator (Optimizer)",
    "📊 Deep Report + AI SWOT",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — OPTIMIZER  (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════
with tab_optimizer:
    st.markdown("### Mean-Variance / HRP / CVaR Optimization")
    st.write("Find the optimal asset allocation for a given risk tolerance.")

    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_area(
            "Enter Stock Tickers (comma separated)",
            "AAPL, GOOG, MSFT, TSLA", height=100,
        )
    with col2:
        risk_tolerance = st.slider(
            "Risk Tolerance (0=Safe, 1=Risky)", 0.0, 1.0, 0.5,
            help="0–0.33 → HRP | 0.34–0.66 → Min CVaR | 0.67–1.0 → Max Sharpe",
        )

    if st.button("⚡ Optimize Portfolio", type="primary", key="opt_btn"):
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.warning("Please enter at least one ticker.")
        else:
            with st.spinner("Calculating optimal allocation…"):
                try:
                    response = http_requests.post(
                        f"{API_URL}/optimize/",
                        json={"tickers": tickers, "risk_tolerance": risk_tolerance},
                    )
                    if response.status_code == 200:
                        resp_json = response.json()
                        data = resp_json.get("allocation", {})
                        reasoning = resp_json.get("reasoning", "No analysis available.")

                        st.success("Optimization complete!")
                        st.subheader("Recommended Allocation")

                        col_chart, col_table = st.columns(2)
                        with col_chart:
                            labels = list(data.keys())
                            values = list(data.values())
                            fig = go.Figure(data=[go.Pie(
                                labels=labels, values=values, hole=0.4,
                                marker=dict(colors=["#58a6ff","#3fb950","#bc8cff",
                                                     "#e3b341","#f85149","#79c0ff"]),
                            )])
                            fig.update_layout(
                                margin=dict(t=20, b=20, l=20, r=20),
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        with col_table:
                            df = pd.DataFrame(list(data.items()), columns=["Ticker", "Weight"])
                            df["Weight"] = df["Weight"].apply(lambda x: f"{x:.1%}")
                            st.table(df)

                        st.info(f"**AI Analysis:**\n\n{reasoning}")
                    else:
                        st.error("Optimization failed.")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — DEEP REPORT + AI SWOT
# ═══════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("### 📊 Professional Quant Report  +  🧠 AI SWOT Intelligence")

    # ── Section A: Quant Backtest Report ─────────────────────────────
    if not QUANT_REPORTER_AVAILABLE:
        st.warning("`quant-reporter` not installed. Backtest section unavailable.")
    else:
        st.write("Generate a comprehensive HTML report — backtesting, "
                 "efficient frontier, and Monte Carlo simulations.")

        with st.expander("📐 Report Configuration", expanded=True):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                repo_tickers = st.text_input("Portfolio Tickers", "AAPL, MSFT, SPY, QQQ",
                                             key="repo_tickers_input")
                benchmark = st.text_input("Benchmark Ticker", "SPY", key="bench_input")
            with col_in2:
                default_start = datetime.now() - timedelta(days=365*2)
                start_date = st.date_input("Training Start Date", default_start,
                                           key="dr_start")
                end_date   = st.date_input("Training End Date",
                                           datetime.now() - timedelta(days=90),
                                           key="dr_end")

        if st.button("🚀 Generate Quant Report", key="gen_report_btn"):
            r_tickers = [t.strip() for t in repo_tickers.split(",") if t.strip()]
            if not r_tickers:
                st.warning("Enter tickers.")
            else:
                weight = 1.0 / len(r_tickers)
                portfolio_dict = {t: weight for t in r_tickers}
                with st.spinner("Running Monte Carlo & Walk-Forward Analysis… (~30s)"):
                    try:
                        report_file = os.path.abspath("portfolio_report.html")
                        qr.create_combined_report(
                            portfolio_dict=portfolio_dict,
                            benchmark_ticker=benchmark,
                            train_start=start_date.strftime("%Y-%m-%d"),
                            train_end=end_date.strftime("%Y-%m-%d"),
                            filename=report_file,
                        )
                        if not os.path.exists(report_file):
                            raise FileNotFoundError(f"Report not created at {report_file}")

                        st.success(f"Report generated at `{report_file}`")
                        with open(report_file, "r", encoding="utf-8") as f:
                            html_content = f.read()

                        st.info("Fetching AI SWOT analyses for portfolio tickers...")
                        swot_outputs = {}
                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(r_tickers))) as executor:
                            future_to_ticker = {executor.submit(call_swot_api, t): t for t in r_tickers}
                            for future in concurrent.futures.as_completed(future_to_ticker):
                                t = future_to_ticker[future]
                                try:
                                    swot_outputs[t] = future.result()
                                except Exception as e:
                                    swot_outputs[t] = {"error": str(e)}

                        swot_html_blocks = []
                        for t in r_tickers:
                            swot_html_blocks.append(build_swot_html_for_report(t, swot_outputs.get(t, {})))
                        
                        swot_section_html = f'''
                        <hr style="margin: 60px 0 40px 0; border: 1px solid #30363d;">
                        <h1 style="font-family: sans-serif; color: #e6edf3; text-align: center; margin-bottom: 40px;">AI SWOT Intelligence</h1>
                        <div style="max-width: 1200px; margin: 0 auto;">
                            {"".join(swot_html_blocks)}
                        </div>
                        '''
                        
                        if "</body>" in html_content:
                            html_content = html_content.replace("</body>", swot_section_html + "\n</body>")
                        else:
                            html_content += swot_section_html
                            
                        with open(report_file, "w", encoding="utf-8") as f:
                            f.write(html_content)

                        st.success("AI SWOT injected successfully.")

                        st.download_button(
                            "⬇️ Download HTML Report (with SWOT)", html_content,
                            "My_Quant_Report.html", mime="text/html",
                        )
                        st.divider()
                        st.subheader("Report Preview")
                        components.html(html_content, height=800, scrolling=True)

                    except Exception as e:
                        st.error(f"Report generation failed: {e}")

