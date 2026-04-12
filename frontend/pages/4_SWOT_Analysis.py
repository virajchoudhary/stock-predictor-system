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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── quadrant cards ─────────────────────────────── */
.swot-card {
    border-radius: 16px;
    padding: 22px 24px;
    min-height: 260px;
    box-shadow: 0 6px 30px rgba(0,0,0,0.35);
    transition: transform 0.25s, box-shadow 0.25s;
}
.swot-card:hover { transform:translateY(-3px); box-shadow:0 10px 40px rgba(0,0,0,0.5); }

.swot-S  { background: linear-gradient(145deg,#0a1f0f,#0d2f18); border:1px solid #2a6b3a; }
.swot-W  { background: linear-gradient(145deg,#1f1505,#2d1e07); border:1px solid #7a5c1e; }
.swot-O  { background: linear-gradient(145deg,#040f1f,#081a35); border:1px solid #1a4a8a; }
.swot-T  { background: linear-gradient(145deg,#1f0507,#350a0d); border:1px solid #8a1a1f; }

.swot-header { font-size:1.15rem; font-weight:800; margin-bottom:14px; letter-spacing:0.5px; }
.swot-S .swot-header { color:#4ade80; }
.swot-W .swot-header { color:#fbbf24; }
.swot-O .swot-header { color:#60a5fa; }
.swot-T .swot-header { color:#f87171; }

.swot-point {
    padding: 10px 12px;
    border-radius: 9px;
    margin-bottom: 10px;
    font-size: 0.875rem;
    line-height: 1.55;
}
.swot-S .swot-point { background:rgba(74,222,128,0.07); border-left:3px solid #4ade80; color:#dcfce7; }
.swot-W .swot-point { background:rgba(251,191,36,0.07); border-left:3px solid #fbbf24; color:#fef3c7; }
.swot-O .swot-point { background:rgba(96,165,250,0.07); border-left:3px solid #60a5fa; color:#dbeafe; }
.swot-T .swot-point { background:rgba(248,113,113,0.07); border-left:3px solid #f87171; color:#fee2e2; }

.swot-evidence { font-size:0.75rem; color:#94a3b8; margin-top:4px; font-style:italic; }
.swot-source   { font-size:0.7rem; color:#64748b; margin-top:2px; }

/* ── rating badge ───────────────────────────────── */
.rating-sb  { background:#052e16;color:#4ade80;padding:8px 22px;border-radius:10px;font-weight:800;font-size:1.1rem;border:1px solid #4ade80;display:inline-block; }
.rating-b   { background:#0a2215;color:#86efac;padding:8px 22px;border-radius:10px;font-weight:800;font-size:1.1rem;border:1px solid #86efac;display:inline-block; }
.rating-h   { background:#1c1407;color:#fbbf24;padding:8px 22px;border-radius:10px;font-weight:800;font-size:1.1rem;border:1px solid #fbbf24;display:inline-block; }
.rating-s   { background:#2d0a0c;color:#f87171;padding:8px 22px;border-radius:10px;font-weight:800;font-size:1.1rem;border:1px solid #f87171;display:inline-block; }
.rating-ss  { background:#1a0507;color:#ef4444;padding:8px 22px;border-radius:10px;font-weight:800;font-size:1.1rem;border:1px solid #ef4444;display:inline-block; }

/* ── metric cards ───────────────────────────────── */
div[data-testid="stMetric"] {
    background:linear-gradient(135deg,#0d1117,#161b22);
    border:1px solid #21262d; border-radius:12px;
    padding:16px 20px; box-shadow:0 4px 18px rgba(0,0,0,0.35);
}
div[data-testid="stMetric"] label { color:#8b949e!important; font-size:0.75rem!important; letter-spacing:0.5px; text-transform:uppercase; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#e6edf3!important; font-weight:700!important; }

/* ── news card ──────────────────────────────────── */
.news-card {
    background:#0d1117; border:1px solid #21262d; border-radius:10px;
    padding:14px 16px; margin-bottom:10px;
    transition: border-color 0.2s;
}
.news-card:hover { border-color:#388bfd44; }
.news-title { font-size:0.9rem; font-weight:600; color:#e6edf3; }
.news-meta  { font-size:0.74rem; color:#8b949e; margin-top:4px; }
.news-summary { font-size:0.8rem; color:#64748b; margin-top:5px; }

/* ── section title ──────────────────────────────── */
.sec-title { font-size:1rem; font-weight:700; color:#e6edf3; border-left:3px solid #388bfd; padding-left:10px; margin:20px 0 12px 0; }

/* ── peer table ─────────────────────────────────── */
.stDataFrame { border-radius:10px; overflow:hidden; }

/* ── executive box ──────────────────────────────── */
.exec-box {
    background:linear-gradient(135deg,#040d1a,#071527);
    border:1px solid #1a3a5c; border-radius:14px;
    padding:20px 24px; margin:12px 0;
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
<div style="padding:0 0 6px 0;">
<h1 style="
    background:linear-gradient(90deg,#4ade80,#60a5fa,#f59e0b,#f87171);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-weight:900; font-size:2.1rem; margin:0; letter-spacing:-0.5px;">
    SWOT Intelligence Engine
</h1>
<p style="color:#8b949e; margin:4px 0 0 0; font-size:0.95rem;">
    Real-time data · Groq LLM synthesis · India (NSE/BSE ₹) &amp; US (NYSE/NASDAQ $)
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
            if st.button(label, key=f"ex_{ticker}", use_container_width=True):
                st.session_state["swot_prefill"] = ticker
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
                         use_container_width=True, key="swot_run")

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
            st.plotly_chart(chart_fig, use_container_width=True)

    # ── SWOT QUADRANTS ────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <h2 style="background:linear-gradient(90deg,#4ade80,#60a5fa,#f59e0b,#f87171);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-weight:900;font-size:1.6rem;margin:0 0 16px 0;">
        SWOT Analysis
    </h2>
    """, unsafe_allow_html=True)

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
        st.dataframe(peer_df, use_container_width=True, hide_index=True)

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
<div style="margin-top:20px; display:flex; justify-content:center; gap:20px; flex-wrap:wrap;">
<span style="background:#0d2818;color:#4ade80;padding:6px 16px;border-radius:8px;font-size:0.85rem;border:1px solid #4ade80;">RELIANCE.NS</span>
<span style="background:#0d2818;color:#4ade80;padding:6px 16px;border-radius:8px;font-size:0.85rem;border:1px solid #4ade80;">TCS.NS</span>
<span style="background:#040f1f;color:#60a5fa;padding:6px 16px;border-radius:8px;font-size:0.85rem;border:1px solid #60a5fa;">AAPL</span>
<span style="background:#040f1f;color:#60a5fa;padding:6px 16px;border-radius:8px;font-size:0.85rem;border:1px solid #60a5fa;">NVDA</span>
</div>
</div>""", unsafe_allow_html=True)

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
