"""
SWOT Analysis - Stock Price Predictor
=====================================
Real-time, data-driven SWOT analysis for Indian and US stocks.
Fetches live fundamentals, news, and peer data → Groq LLM synthesis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests as http_requests
import sys
import os
import math
from datetime import datetime

# ── backend path ──────────────────────────────────────────────────────
_backend = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
if os.path.isdir(_backend) and _backend not in sys.path:
    sys.path.insert(0, _backend)

API_URL = "http://127.0.0.1:8000/api"

# ── page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SWOT Analysis - Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS — Premium dark analyst terminal
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Roboto+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── quadrant cards ─────────────────────────────── */
.swot-card {
    border-radius: 2px;
    padding: 18px 20px;
    min-height: 240px;
    background: #1E222D;
}

.swot-S  { border: 1px solid #2A2E39; border-top: 2px solid #26A69A; }
.swot-W  { border: 1px solid #2A2E39; border-top: 2px solid #F7A600; }
.swot-O  { border: 1px solid #2A2E39; border-top: 2px solid #2196F3; }
.swot-T  { border: 1px solid #2A2E39; border-top: 2px solid #EF5350; }

.swot-header { font-family:'Roboto Mono',monospace; font-size:0.6rem; font-weight:500; margin-bottom:12px; letter-spacing:0.15em; text-transform:uppercase; }
.swot-S .swot-header { color:#26A69A; }
.swot-W .swot-header { color:#F7A600; }
.swot-O .swot-header { color:#2196F3; }
.swot-T .swot-header { color:#EF5350; }

.swot-point {
    padding: 8px 10px;
    border-radius: 2px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    line-height: 1.5;
    background: rgba(255,255,255,0.03);
}
.swot-S .swot-point { border-left:2px solid #26A69A; color:#D1D4DC; }
.swot-W .swot-point { border-left:2px solid #F7A600; color:#D1D4DC; }
.swot-O .swot-point { border-left:2px solid #2196F3; color:#D1D4DC; }
.swot-T .swot-point { border-left:2px solid #EF5350; color:#D1D4DC; }

.swot-evidence { font-size:0.72rem; color:#787B86; margin-top:3px; }
.swot-source   { font-size:0.68rem; color:#787B86; margin-top:2px; }

/* ── rating badge ───────────────────────────────── */
.rating-sb  { background:#1E222D;color:#26A69A;padding:6px 18px;border-radius:2px;font-weight:600;font-size:0.85rem;border:1px solid #26A69A;display:inline-block;font-family:'Roboto Mono',monospace; }
.rating-b   { background:#1E222D;color:#26A69A;padding:6px 18px;border-radius:2px;font-weight:600;font-size:0.85rem;border:1px solid #26A69A;display:inline-block;font-family:'Roboto Mono',monospace; }
.rating-h   { background:#1E222D;color:#F7A600;padding:6px 18px;border-radius:2px;font-weight:600;font-size:0.85rem;border:1px solid #F7A600;display:inline-block;font-family:'Roboto Mono',monospace; }
.rating-s   { background:#1E222D;color:#EF5350;padding:6px 18px;border-radius:2px;font-weight:600;font-size:0.85rem;border:1px solid #EF5350;display:inline-block;font-family:'Roboto Mono',monospace; }
.rating-ss  { background:#1E222D;color:#EF5350;padding:6px 18px;border-radius:2px;font-weight:600;font-size:0.85rem;border:1px solid #EF5350;display:inline-block;font-family:'Roboto Mono',monospace; }

/* ── metric cards ───────────────────────────────── */
div[data-testid="stMetric"] {
    background: #1E222D;
    border: 1px solid #2A2E39;
    border-radius: 2px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label { color:#787B86!important; font-size:0.57rem!important; letter-spacing:0.15em; text-transform:uppercase; font-family:'Roboto Mono',monospace!important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#D1D4DC!important; font-weight:500!important; }

/* ── news card ──────────────────────────────────── */
.news-card {
    background:#1E222D; border:1px solid #2A2E39; border-radius:2px;
    padding:12px 14px; margin-bottom:8px;
}
.news-card:hover { border-color:#2196F3; }
.news-title { font-size:0.85rem; font-weight:500; color:#D1D4DC; }
.news-meta  { font-size:0.72rem; color:#787B86; margin-top:3px; font-family:'Roboto Mono',monospace; }
.news-summary { font-size:0.78rem; color:#787B86; margin-top:4px; }

/* ── section title ──────────────────────────────── */
.sec-title { font-family:'Roboto Mono',monospace; font-size:0.57rem; font-weight:400; color:#787B86; border-left:2px solid #2196F3; padding-left:8px; margin:16px 0 10px 0; letter-spacing:0.15em; text-transform:uppercase; }

/* ── peer table ─────────────────────────────────── */
.stDataFrame { border-radius:2px; overflow:hidden; }

/* ── executive box ──────────────────────────────── */
.exec-box {
    background: #1E222D;
    border: 1px solid #2A2E39; border-radius:2px;
    padding:16px 20px; margin:10px 0;
}
.exec-text { color:#93c5fd; font-size:1rem; line-height:1.7; }

/* ── flag chips ─────────────────────────────────── */
.flag-green  { background:#052e16;color:#4ade80;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700; display:inline-block; margin-left:6px; }
.flag-orange { background:#1c1407;color:#fbbf24;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700; display:inline-block; margin-left:6px; }
.flag-blue   { background:#040f1f;color:#60a5fa;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700; display:inline-block; margin-left:6px; }
.flag-red    { background:#2d0a0c;color:#f87171;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700; display:inline-block; margin-left:6px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def rating_badge(rating: str) -> str:
    r = rating.upper().strip()
    cls = {"STRONG BUY": "rating-sb", "BUY": "rating-b",
           "HOLD": "rating-h", "SELL": "rating-s", "STRONG SELL": "rating-ss"}
    icon = {"STRONG BUY": "⬆⬆", "BUY": "⬆", "HOLD": "◆", "SELL": "⬇", "STRONG SELL": "⬇⬇"}
    for key in cls:
        if key in r:
            return f'<span class="{cls[key]}">{icon[key]} {key}</span>'
    return f'<span class="rating-h">◆ {rating}</span>'


def flag_chip(flag: str) -> str:
    mapping = {"GREEN": ("flag-green", "Strength"), "ORANGE": ("flag-orange", "Caution"),
               "BLUE": ("flag-blue", "Opportunity"), "RED": ("flag-red", "Risk")}
    cls, label = mapping.get(flag.upper(), ("flag-blue", flag))
    return f'<span class="{cls}">{label}</span>'


def render_swot_card(quadrant_key: str, title: str, emoji: str, points: list):
    css_cls = {"S": "swot-S", "W": "swot-W", "O": "swot-O", "T": "swot-T"}[quadrant_key]

    points_html = ""
    for p in points:
        point = p.get("point", "")
        evidence = p.get("evidence", "")
        source = p.get("source", "")
        flag = p.get("flag", "")
        chip = flag_chip(flag) if flag else ""
        points_html += f"""<div class="swot-point">
<strong>{point}</strong>{chip}
{f'<div class="swot-evidence">Evidence: {evidence}</div>' if evidence else ''}
{f'<div class="swot-source">Source: {source}</div>' if source else ''}
</div>
"""

    prefix = f"{emoji} " if emoji else ""
    st.markdown(f"""<div class="swot-card {css_cls}">
<div class="swot-header">{prefix}{title}</div>
{points_html}
</div>""", unsafe_allow_html=True)


def render_mini_chart(price_metrics: dict, company_name: str, currency: str):
    dates = price_metrics.get("dates", [])
    prices = price_metrics.get("close_prices", [])
    if not dates or not prices:
        return None

    color = "#4ade80" if prices[-1] >= prices[0] else "#f87171"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        fill="tozeroy",
        fillcolor=f"rgba({','.join(str(int(int(color[i:i+2],16))) for i in (1,3,5))},0.08)",
        line=dict(color=color, width=2),
        hovertemplate=f"{currency}%{{y:,.2f}}<extra></extra>",
    ))
    pct_change = (prices[-1] / prices[0] - 1) * 100 if prices[0] else 0
    fig.update_layout(
        title=dict(text=f"{company_name} — 60-Day Price ({'+' if pct_change >= 0 else ''}{pct_change:.2f}%)",
                   font=dict(color="#e6edf3", size=13)),
        height=260, template="plotly_dark",
        plot_bgcolor="rgba(13,17,23,0.9)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=45, b=30),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#21262d", tickformat=",.0f"),
        font=dict(family="Inter"),
    )
    return fig


def render_fundamentals_grid(fund: dict, pm: dict):
    curr = fund.get("currency", "$")
    metrics = [
        ("Market Cap", fund.get("market_cap_fmt", "N/A"), None),
        ("P/E Ratio", str(round(fund["pe_ratio"], 1)) if fund.get("pe_ratio") else "N/A", None),
        ("P/B Ratio", str(round(fund["pb_ratio"], 1)) if fund.get("pb_ratio") else "N/A", None),
        ("EPS (TTM)", f"{curr}{fund['eps']:.2f}" if fund.get("eps") else "N/A", None),
        ("ROE", f"{fund['roe']*100:.1f}%" if fund.get("roe") else "N/A", None),
        ("Revenue", fund.get("revenue_fmt", "N/A"), None),
        ("Net Income", fund.get("net_income_fmt", "N/A"), None),
        ("Profit Margin", f"{fund['profit_margin']*100:.1f}%" if fund.get("profit_margin") else "N/A", None),
        ("Revenue Growth", f"{fund['revenue_growth']*100:.1f}%" if fund.get("revenue_growth") else "N/A",
         f"+{fund['revenue_growth']*100:.1f}%" if fund.get("revenue_growth", 0) and fund["revenue_growth"] > 0 else None),
        ("Debt/Equity", str(round(fund["debt_equity"], 1)) if fund.get("debt_equity") else "N/A", None),
        ("Free Cash Flow", fund.get("free_cashflow_fmt", "N/A"), None),
        ("Beta", str(round(fund["beta"], 2)) if fund.get("beta") else "N/A", None),
        ("52W High", f"{curr}{fund['week52_high']:,.2f}" if fund.get("week52_high") else "N/A", None),
        ("52W Low", f"{curr}{fund['week52_low']:,.2f}" if fund.get("week52_low") else "N/A", None),
        ("Analyst Target", f"{curr}{fund['analyst_target']:,.2f}" if fund.get("analyst_target") else "N/A", None),
        ("Recommendation", (fund.get("recommendation") or "N/A").upper(), None),
    ]
    # price metrics extras
    if pm:
        metrics += [
            ("52W Perf.", f"{pm.get('perf_52w', 'N/A')}%", f"+{pm['perf_52w']:.1f}%" if pm.get("perf_52w", 0) > 0 else None),
            ("30D Perf.", f"{pm.get('perf_30d', 'N/A')}%", None),
            ("Ann. Vol.", f"{pm.get('ann_volatility_pct', 'N/A')}%", None),
            ("RSI (14)", str(pm.get("rsi_14", "N/A")), None),
        ]

    cols = st.columns(4)
    for i, (label, val, delta) in enumerate(metrics):
        with cols[i % 4]:
            st.metric(label, val, delta)


# ═══════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="border-bottom:1px solid #2A2E39;padding-bottom:0.875rem;margin-bottom:1.5rem;">
    <span style="font-family:'Roboto Mono',monospace;font-size:0.57rem;letter-spacing:0.18em;
    color:#26A69A;text-transform:uppercase;">Fundamental Research</span>
    <h1 style="font-family:'Inter',sans-serif;font-size:1.35rem;font-weight:500;
    color:#D1D4DC;margin:0.25rem 0 0 0;letter-spacing:-0.02em;">SWOT Analysis</h1>
    <p style="font-family:'Inter',sans-serif;color:#787B86;margin:0.35rem 0 0 0;font-size:0.82rem;">
        Real-time data &nbsp;&middot;&nbsp; LLM synthesis &nbsp;&middot;&nbsp; India (NSE/BSE) &amp; US (NYSE/NASDAQ)
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# INPUT SECTION
# ═══════════════════════════════════════════════════════════════════════
# Use a separate prefill key so quick-example buttons don't conflict
# with the already-instantiated text_input widget key.
if "swot_prefill" not in st.session_state:
    st.session_state["swot_prefill"] = ""
if "swot_auto_run" not in st.session_state:
    st.session_state["swot_auto_run"] = False

col_inp, col_ex = st.columns([3, 2], gap="large")

with col_ex:
    st.markdown('<div class="sec-title">Quick Examples</div>', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    examples = [
        ("Reliance", "RELIANCE.NS"),
        ("TCS", "TCS.NS"),
        ("HDFC Bank", "HDFCBANK.NS"),
        ("Apple", "AAPL"),
        ("Nvidia", "NVDA"),
        ("Tesla", "TSLA"),
    ]
    for i, (label, ticker) in enumerate(examples):
        with ex_cols[i % 3]:
            if st.button(label, key=f"ex_{ticker}", width='stretch'):
                st.session_state["swot_prefill"] = ticker
                st.session_state["swot_auto_run"] = True
                st.rerun()

with col_inp:
    st.markdown('<div class="sec-title">Company / Ticker</div>', unsafe_allow_html=True)
    company_query = st.text_input(
        "Enter Company Name or Ticker Symbol",
        value=st.session_state.get("swot_prefill", ""),
        placeholder="e.g. Reliance, AAPL, TCS, Tesla, HINDALCO.NS …",
        key="swot_query",
        label_visibility="collapsed",
    )
    st.caption("Supports: Indian companies (NSE/BSE) and US stocks. Enter name or ticker symbol.")

run_clicked = st.button("Generate SWOT Analysis", type="primary",
                         width='stretch', key="swot_run")

# Auto-run when a quick example was clicked
if st.session_state.get("swot_auto_run") and company_query.strip():
    run_clicked = True
    st.session_state["swot_auto_run"] = False

# ── Session state for caching results ────────────────────────────────
if "swot_result" not in st.session_state:
    st.session_state["swot_result"] = None

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS EXECUTION
# ═══════════════════════════════════════════════════════════════════════
if run_clicked and company_query.strip():
    with st.spinner(f"Fetching real-time data for **{company_query}** ...  "
                    "*(Collecting fundamentals, news, peers → Groq LLM synthesis)*"):
        try:
            resp = http_requests.post(
                f"{API_URL}/swot/",
                json={"query": company_query.strip()},
                timeout=90,
            )
            if resp.status_code == 200:
                st.session_state["swot_result"] = resp.json()
            else:
                err = resp.json().get("error", "Unknown error")
                st.error(f"API Error: {err}")
                st.session_state["swot_result"] = None
        except http_requests.exceptions.ConnectionError:
            # Fallback: call backend directly (no Django running)
            try:
                sys.path.insert(0, _backend)
                import django
                os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantvision.settings")
                # Try direct import
                from api.swot_service import run_swot_analysis
                result = run_swot_analysis(company_query.strip())
                st.session_state["swot_result"] = result
            except Exception as e_direct:
                st.error(f"Backend connection failed. Start Django backend or check connection.\n\nError: {e_direct}")
                st.session_state["swot_result"] = None
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state["swot_result"] = None

elif run_clicked and not company_query.strip():
    st.warning("Please enter a company name or ticker symbol.")

# ═══════════════════════════════════════════════════════════════════════
# RESULT RENDERING
# ═══════════════════════════════════════════════════════════════════════
data = st.session_state.get("swot_result")

if data and "error" not in data:
    fund = data.get("fundamentals", {})
    pm = data.get("price_metrics", {})
    news = data.get("news", [])
    peers = data.get("peers", [])
    swot = data.get("swot", {})
    curr = data.get("currency", "$")
    company_name = data.get("company_name", data.get("ticker", "Company"))
    ticker_sym = data.get("ticker", "")
    market = data.get("market", "US")
    generated_at = data.get("generated_at", "")

    # ── COMPANY HEADER ────────────────────────────────────────────────
    h1, h2, h3 = st.columns([3, 1.5, 1.5])
    with h1:
        mkt_flag = "NSE/BSE · ₹" if market == "IN" else "NYSE/NASDAQ · $"
        st.markdown(f"""<div style="padding:16px 20px; background:#0d1117; border:1px solid #21262d; border-radius:12px;">
<div style="font-size:1.5rem;font-weight:800;color:#e6edf3;">{company_name}</div>
<div style="color:#8b949e;font-size:0.88rem;margin-top:4px;">
{ticker_sym} &nbsp;·&nbsp; {mkt_flag} &nbsp;·&nbsp;
{fund.get('sector','N/A')} &nbsp;·&nbsp; {fund.get('industry','N/A')}
</div>
<div style="color:#484f58;font-size:0.75rem;margin-top:6px;">
Generated: {generated_at}
</div>
</div>""", unsafe_allow_html=True)

    with h2:
        price_val = fund.get("price")
        day_chg = fund.get("day_change_pct")
        if price_val:
            st.metric(
                "Current Price",
                f"{curr}{price_val:,.2f}",
                f"{day_chg:+.2f}%" if day_chg else None,
            )

    with h3:
        rec = fund.get("recommendation", "N/A")
        target = fund.get("analyst_target")
        if target:
            st.metric("Analyst Target", f"{curr}{target:,.2f}")
        st.markdown(f"**Consensus:** `{rec.upper()}`")

    # ── INVESTMENT RATING ─────────────────────────────────────────────
    if swot.get("investment_rating"):
        st.divider()
        r1, r2, r3 = st.columns([2, 3, 2])
        with r1:
            st.markdown('<div class="sec-title">Investment Rating</div>', unsafe_allow_html=True)
            st.markdown(rating_badge(swot["investment_rating"]), unsafe_allow_html=True)
        with r2:
            if swot.get("executive_summary"):
                st.markdown('<div class="exec-box"><div class="exec-text">' +
                            swot["executive_summary"] + '</div></div>', unsafe_allow_html=True)
        with r3:
            if swot.get("key_risk"):
                st.error(f"Key Risk: {swot['key_risk']}")
            if swot.get("key_opportunity"):
                st.success(f"Key Opportunity: {swot['key_opportunity']}")

    # ── FUNDAMENTALS GRID ─────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="sec-title">Financial Snapshot</div>', unsafe_allow_html=True)
    render_fundamentals_grid(fund, pm)

    # ── PRICE CHART ───────────────────────────────────────────────────
    if pm:
        chart_fig = render_mini_chart(pm, company_name, curr)
        if chart_fig:
            st.plotly_chart(chart_fig, width='stretch')

    # ── SWOT QUADRANTS ────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="sec-title">SWOT Quadrants</div>', unsafe_allow_html=True)

    if "error" in swot:
        st.error(f"SWOT generation error: {swot['error']}")
        if swot.get("raw"):
            with st.expander("Raw LLM Output"):
                st.code(swot["raw"])
    else:
        sw1, sw2 = st.columns(2, gap="large")
        with sw1:
            render_swot_card("S", "STRENGTHS", "", swot.get("strengths", []))
            st.markdown("<br>", unsafe_allow_html=True)
            render_swot_card("O", "OPPORTUNITIES", "", swot.get("opportunities", []))
        with sw2:
            render_swot_card("W", "WEAKNESSES", "", swot.get("weaknesses", []))
            st.markdown("<br>", unsafe_allow_html=True)
            render_swot_card("T", "THREATS", "", swot.get("threats", []))

    # ── PEER COMPARISON ───────────────────────────────────────────────
    if peers:
        st.divider()
        st.markdown('<div class="sec-title">Competitor Benchmarking</div>', unsafe_allow_html=True)
        peer_df = pd.DataFrame(peers)
        st.dataframe(peer_df, width='stretch', hide_index=True)

    # ── NEWS FEED ─────────────────────────────────────────────────────
    if news:
        st.divider()
        st.markdown(f'<div class="sec-title">Latest News ({len(news)} items)</div>',
                    unsafe_allow_html=True)
        for n in news[:10]:
            link = n.get("link", "#")
            title = n.get("title", "")
            source = n.get("source", "Unknown")
            date = n.get("date", "")
            summary = n.get("summary", "")
            st.markdown(f"""<div class="news-card">
<div class="news-title">
<a href="{link}" target="_blank" style="color:#e6edf3;text-decoration:none;">
{title}
</a>
</div>
<div class="news-meta">{source} &nbsp;·&nbsp; {date}</div>
{f'<div class="news-summary">{summary}</div>' if summary else ''}
</div>""", unsafe_allow_html=True)

    # ── COMPANY DESCRIPTION ───────────────────────────────────────────
    desc = fund.get("description", "")
    if desc:
        with st.expander("Company Overview"):
            st.markdown(f'<p style="color:#94a3b8;line-height:1.8;font-size:0.9rem;">{desc}</p>',
                        unsafe_allow_html=True)
            info_cols = st.columns(3)
            with info_cols[0]:
                st.markdown(f"**Country:** {fund.get('country', 'N/A')}")
                st.markdown(f"**Exchange:** {fund.get('exchange', 'N/A')}")
            with info_cols[1]:
                st.markdown(f"**Employees:** {fund.get('employees', 'N/A'):,}" if isinstance(fund.get('employees'), int) else f"**Employees:** {fund.get('employees', 'N/A')}")
                st.markdown(f"**Dividend Yield:** {fund['dividend_yield']*100:.2f}%" if fund.get("dividend_yield") else "**Dividend Yield:** N/A")
            with info_cols[2]:
                website = fund.get("website", "")
                if website and website != "N/A":
                    st.markdown(f"**Website:** [{website}]({website})")

    # ── DISCLAIMER ───────────────────────────────────────────────────
    st.divider()
    st.caption(
        "**Disclaimer:** This analysis is generated using real-time market data and AI. "
        "It is for educational and informational purposes only. "
        "Not financial advice. Always conduct your own due diligence before investing."
    )

elif data and "error" in data:
    st.error(f"Could not generate SWOT: {data['error']}")

else:
    # ── PLACEHOLDER when nothing has been run ─────────────────────────
    st.markdown("""<div style="text-align:center; padding:60px 20px; color:#484f58;">
<div style="font-size:1.2rem; color:#6e7681; margin-top:12px; font-weight:600;">
Enter a company name or ticker and click "Generate SWOT Analysis"
</div>
<div style="font-size:0.9rem; color:#484f58; margin-top:8px; max-width:500px; margin-left:auto; margin-right:auto; line-height:1.7;">
The engine will fetch live financial data, recent news, peer comparisons,
and use Groq AI (Llama 3.3 70B) to synthesize a professional SWOT analysis.
</div>
</div>""", unsafe_allow_html=True)
    placeholder_cols = st.columns(4)
    placeholder_tickers = ["RELIANCE.NS", "TCS.NS", "AAPL", "NVDA"]
    for i, pticker in enumerate(placeholder_tickers):
        with placeholder_cols[i]:
            if st.button(pticker, key=f"ph_{pticker}", width='stretch'):
                    st.session_state["swot_prefill"] = pticker
                    st.session_state["swot_auto_run"] = True
                    st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data Sources")
    st.markdown("""
**Live Data Feeds:**
- yfinance (30+ financial metrics)
- Yahoo Finance RSS news feed
- Analyst recommendations
- Peer / competitor data

**AI Engine:**
- Groq · Llama 3.3 70B
- Investment-grade prompt
- Structured JSON output

---

**Supported Markets:**
| Market | Currency |
|--|--|
| NSE / BSE | ₹ INR |
| NYSE / NASDAQ | $ USD |

---

**Ticker Examples:**
```
India:  RELIANCE.NS
        TCS.NS
        INFY.NS
        HDFCBANK.NS
        TATAMOTORS.NS

US:     AAPL  MSFT
        NVDA  TSLA
        JPM   GOOGL
```
    """)
    st.divider()
    st.caption("Educational use only. Not financial advice.")
