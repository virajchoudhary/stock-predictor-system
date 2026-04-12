"""
Black–Scholes Options Analyzer — v2
=====================================
Supports Indian (NSE/BSE) and US (NYSE/NASDAQ) markets.
Auto-detects market, displays correct currency, uses country-specific
risk-free rates, and shows confidence scores for every signal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import sys
import os
import io
from datetime import datetime, timedelta

# ── path setup ─────────────────────────────────────────────────────────
_backend = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
if os.path.isdir(_backend) and _backend not in sys.path:
    sys.path.insert(0, _backend)

from api.black_scholes import (
    detect_market, get_currency_symbol, get_default_rate,
    format_price, historical_volatility,
    bs_call_price, bs_put_price, bs_price,
    greeks, implied_volatility, valuation_signal,
    confidence_score, analyze_option,
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: box-shadow 0.2s;
}
div[data-testid="stMetric"]:hover { box-shadow: 0 6px 28px rgba(56,139,253,0.12); }
div[data-testid="stMetric"] label { color:#8b949e !important; font-size:0.78rem !important; letter-spacing:0.6px; text-transform:uppercase; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#e6edf3 !important; font-weight:700 !important; font-size:1.4rem !important; }

/* signal badges */
.badge-buy  { background:#0d2818; color:#3fb950; padding:7px 20px; border-radius:9px; font-weight:700; border:1px solid #3fb950; font-size:1rem; }
.badge-sell { background:#2d1117; color:#f85149; padding:7px 20px; border-radius:9px; font-weight:700; border:1px solid #f85149; font-size:1rem; }
.badge-hold { background:#2d2100; color:#e3b341; padding:7px 20px; border-radius:9px; font-weight:700; border:1px solid #e3b341; font-size:1rem; }

/* market tag */
.tag-india { background:#0d2d09; color:#56d364; padding:3px 12px; border-radius:6px; font-size:0.8rem; font-weight:600; border:1px solid #56d364; }
.tag-us    { background:#0d1f2d; color:#58a6ff; padding:3px 12px; border-radius:6px; font-size:0.8rem; font-weight:600; border:1px solid #58a6ff; }

/* greek cards */
.greek-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:12px 0; }
.greek-card {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d; border-radius: 12px; padding: 16px 10px;
    text-align: center; transition: all 0.2s;
}
.greek-card:hover { border-color:#58a6ff44; transform:translateY(-2px); box-shadow:0 8px 24px rgba(88,166,255,0.1); }
.greek-sym  { font-size:1.5rem; font-weight:800; color:#58a6ff; }
.greek-name { font-size:0.7rem; color:#8b949e; letter-spacing:0.8px; text-transform:uppercase; margin:3px 0; }
.greek-val  { font-size:1.15rem; font-weight:700; color:#e6edf3; }
.greek-desc { font-size:0.65rem; color:#484f58; margin-top:4px; }

/* confidence bar */
.conf-bar-wrap { background:#21262d; border-radius:6px; height:10px; overflow:hidden; margin:4px 0; }
.conf-bar-fill { height:100%; border-radius:6px; transition:width 0.6s ease; }

/* section title */
.sec-title {
    font-size:1.1rem; font-weight:700; color:#e6edf3;
    border-left: 3px solid #388bfd; padding-left: 10px;
    margin: 20px 0 12px 0;
}

/* info banner */
.info-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #0d2134 100%);
    border: 1px solid #1f4068; border-radius:10px; padding:14px 18px; margin:8px 0;
    font-size:0.88rem; color:#a8c5e0;
}

/* table style override */
.stDataFrame { border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════

def render_signal_badge(signal: str, reasoning: str, deviation: float = 0.0):
    css = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[signal]
    icon = {"BUY": "▲", "SELL": "▼", "HOLD": "●"}[signal]
    st.markdown(f'<span class="{css}">{icon} {signal}</span>', unsafe_allow_html=True)
    if reasoning:
        st.caption(reasoning)


def render_market_tag(market: str):
    if market == "IN":
        st.markdown('<span class="tag-india">NSE / BSE · INR ₹</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="tag-us">NYSE / NASDAQ · USD $</span>', unsafe_allow_html=True)


def render_confidence(conf: dict):
    score = conf["score"]
    level = conf["level"]
    color = {"High": "#3fb950", "Medium": "#e3b341", "Low": "#f85149"}.get(level, "#8b949e")
    st.markdown(f"""
    <div style="margin:8px 0;">
        <span style="color:{color};font-weight:700;font-size:0.95rem;">
            {'▓' * int(score // 10)}{'░' * (10 - int(score // 10))}
            &nbsp; {score:.0f}/100 Confidence — {level}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_greeks(g: dict, option_type: str):
    ot = option_type.lower()
    selected = g.get(ot, g.get("call", {}))
    data = [
        ("Δ", "Delta",  selected.get("delta", 0),  "∂Price/∂Spot"),
        ("Γ", "Gamma",  selected.get("gamma", 0),  "Rate of Δ change"),
        ("Θ", "Theta",  selected.get("theta", 0),  "Daily time decay"),
        ("ν", "Vega",   selected.get("vega", 0),   "Per 1% vol move"),
        ("ρ", "Rho",    selected.get("rho", 0),    "Per 1% rate move"),
    ]
    cols = st.columns(5)
    for i, (sym, name, val, desc) in enumerate(data):
        with cols[i]:
            color = "#3fb950" if val > 0 else ("#f85149" if val < 0 else "#8b949e")
            st.markdown(f"""
            <div class="greek-card">
                <div class="greek-sym">{sym}</div>
                <div class="greek-name">{name}</div>
                <div class="greek-val" style="color:{color};">{val:+.4f}</div>
                <div class="greek-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def plot_payoff(S: float, K: float, premium: float, option_type: str, currency: str):
    px_range = np.linspace(S * 0.5, S * 1.5, 400)
    if option_type.lower() == "call":
        payoff = np.maximum(px_range - K, 0) - premium
    else:
        payoff = np.maximum(K - px_range, 0) - premium

    breakeven = (K + premium) if option_type.lower() == "call" else (K - premium)
    profit_zone = payoff >= 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=px_range, y=payoff,
        fill='tozeroy',
        fillcolor='rgba(63,185,80,0.08)',
        line=dict(color='#3fb950', width=2.5),
        name='P&L at Expiry',
        hovertemplate=f"{currency}%{{y:.2f}}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#484f58", line_width=1)
    fig.add_vline(x=K, line_dash="dot", line_color="#e3b341", line_width=1.5,
                  annotation_text=f"Strike {currency}{K:,.0f}",
                  annotation_font_color="#e3b341")
    fig.add_vline(x=S, line_dash="dot", line_color="#58a6ff", line_width=1.5,
                  annotation_text=f"Spot {currency}{S:,.0f}",
                  annotation_font_color="#58a6ff")
    fig.add_vline(x=breakeven, line_dash="dash", line_color="#f85149", line_width=1.5,
                  annotation_text=f"B/E {currency}{breakeven:,.2f}",
                  annotation_font_color="#f85149")
    fig.update_layout(
        title=dict(text=f"{'Call' if option_type.lower()=='call' else 'Put'} Payoff Diagram",
                   font=dict(color="#e6edf3", size=14)),
        xaxis_title="Stock Price at Expiry",
        yaxis_title=f"P&L ({currency})",
        template="plotly_dark",
        height=360,
        margin=dict(l=50, r=20, t=50, b=50),
        plot_bgcolor='rgba(13,17,23,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
    )
    return fig


def plot_greeks_sensitivity(S, K, T, r, sigma, q, option_type, currency):
    spot_range = np.linspace(max(S * 0.6, 1), S * 1.4, 150)
    deltas, gammas, thetas, vegas = [], [], [], []
    ot = option_type.lower()

    for s in spot_range:
        try:
            g = greeks(s, K, T, r, sigma, q)
            deltas.append(g[ot]["delta"])
            gammas.append(g[ot]["gamma"])
            thetas.append(g[ot]["theta"])
            vegas.append(g[ot]["vega"])
        except Exception:
            deltas.append(0); gammas.append(0)
            thetas.append(0); vegas.append(0)

    color_map = {"delta": "#58a6ff", "gamma": "#bc8cff", "theta": "#f85149", "vega": "#3fb950"}
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Delta (Δ)", "Gamma (Γ)", "Theta (Θ)/day", "Vega (ν)/1%"),
                        vertical_spacing=0.18, horizontal_spacing=0.12)

    for (row, col), (data, name) in zip(
        [(1,1),(1,2),(2,1),(2,2)],
        [(deltas,"delta"),(gammas,"gamma"),(thetas,"theta"),(vegas,"vega")]
    ):
        fig.add_trace(go.Scatter(
            x=spot_range, y=data, name=name.capitalize(),
            line=dict(color=color_map[name], width=2),
            showlegend=False,
        ), row=row, col=col)
        fig.add_vline(x=S, line_dash="dot", line_color="#e3b341",
                      line_width=1, row=row, col=col)

    fig.update_layout(
        height=430, template="plotly_dark",
        plot_bgcolor='rgba(13,17,23,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#8b949e'),
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text="Greeks vs. Spot Price", font=dict(color="#e6edf3", size=14)),
    )
    return fig


# ─── Indian market: generate synthetic option chain ────────────────────
@st.cache_data(ttl=300)
def build_indian_option_chain(symbol: str, spot: float, hist_vol: float,
                               r: float, expiry_days: int = 28):
    """
    Build a realistic BS-priced option chain for Indian stocks
    (yfinance does not provide options data for .NS / .BO).
    Uses historical vol + Vol skew to simulate realistic premiums.
    """
    T = expiry_days / 365.0
    expiry_str = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")

    # Strike grid: ±12% around ATM in 2.5% increments for large caps
    step_pct = 0.025
    n_strikes = 9
    strikes = [spot * (1 + step_pct * (i - n_strikes // 2))
               for i in range(n_strikes)]
    # Round to nearest 50 for large Indian stocks
    if spot > 500:
        strikes = [round(s / 50) * 50 for s in strikes]
    else:
        strikes = [round(s / 5) * 5 for s in strikes]
    strikes = sorted(set(strikes))

    rows_call, rows_put = [], []
    for K in strikes:
        # Volatility smile: add skew (puts more expensive)
        log_m = math.log(spot / K) if K > 0 else 0
        vol_call = hist_vol * (1 + 0.05 * log_m)     # slight upward skew for calls
        vol_put = hist_vol * (1 - 0.10 * log_m)       # steeper downward skew for puts
        vol_call = max(vol_call, 0.05)
        vol_put = max(vol_put, 0.05)

        call_p = bs_call_price(spot, K, T, r, vol_call)
        put_p = bs_put_price(spot, K, T, r, vol_put)

        rows_call.append({"strike": K, "lastPrice": round(call_p, 2),
                          "impliedVolatility": round(vol_call, 4),
                          "contractSymbol": f"{symbol}_C_{int(K)}"})
        rows_put.append({"strike": K, "lastPrice": round(put_p, 2),
                         "impliedVolatility": round(vol_put, 4),
                         "contractSymbol": f"{symbol}_P_{int(K)}"})

    return pd.DataFrame(rows_call), pd.DataFrame(rows_put), expiry_str, True


@st.cache_data(ttl=300)
def fetch_spot_and_vol(symbol: str):
    """Fetch spot price and compute 30-day historical vol via yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        if hist.empty:
            return None, None, None
        spot = float(hist["Close"].iloc[-1])
        closes = hist["Close"].tolist()
        hvol = historical_volatility(closes, 30)
        info = ticker.info
        div_yield = float(info.get("dividendYield") or 0.0)
        return spot, hvol, div_yield
    except Exception:
        return None, None, None


@st.cache_data(ttl=300)
def fetch_us_options(symbol: str):
    """Fetch live US option chain from yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        exps = ticker.options
        if not exps:
            return None, None, None, None, None
        hist = ticker.history(period="3mo")
        if hist.empty:
            return None, None, None, None, None
        spot = float(hist["Close"].iloc[-1])
        closes = hist["Close"].tolist()
        hvol = historical_volatility(closes, 30)
        expiry = exps[0]
        chain = ticker.option_chain(expiry)
        info = ticker.info
        div_yield = float(info.get("dividendYield") or 0.0)
        return chain.calls, chain.puts, spot, expiry, hvol, div_yield
    except Exception:
        return None, None, None, None, None, None


# ═══════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="border-bottom:1px solid #2A2E39;padding-bottom:0.875rem;margin-bottom:1.5rem;">
    <span style="font-family:'Roboto Mono',monospace;font-size:0.57rem;letter-spacing:0.18em;
    color:#2196F3;text-transform:uppercase;">Derivatives Research</span>
    <h1 style="font-family:'Inter',sans-serif;font-size:1.35rem;font-weight:500;
    color:#D1D4DC;margin:0.25rem 0 0 0;letter-spacing:-0.02em;">Black-Scholes Options Analyzer</h1>
    <p style="font-family:'Inter',sans-serif;color:#787B86;margin:0.35rem 0 0 0;font-size:0.82rem;">
        India (NSE/BSE) &amp; US (NYSE/NASDAQ) &nbsp;&mdash;&nbsp;
        B-S Pricing &nbsp;&middot;&nbsp; Greeks &nbsp;&middot;&nbsp; Implied Vol &nbsp;&middot;&nbsp; Confidence Signals
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab_manual, tab_live, tab_batch, tab_csv = st.tabs([
    "Manual Analysis",
    "Live Market",
    "Batch (Multi-Stock)",
    "CSV Upload",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — MANUAL
# ═══════════════════════════════════════════════════════════════════════
with tab_manual:
    # ── input panel ──────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<div class="sec-title">Market & Asset</div>', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            m_market_choice = st.selectbox("Market", ["Auto-Detect", "India (NSE/BSE)",
                                                       "US (NYSE/NASDAQ)"], key="m_mkt_choice")
        with mc2:
            m_ticker = st.text_input("Ticker (optional, for label)", "AAPL", key="m_ticker")

        if m_market_choice == "Auto-Detect":
            m_market = detect_market(m_ticker)
        elif "India" in m_market_choice:
            m_market = "IN"
        else:
            m_market = "US"

        m_currency = get_currency_symbol(m_market)
        render_market_tag(m_market)

        st.markdown('<div class="sec-title">Option Parameters</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            m_S = st.number_input(f"Spot Price ({m_currency})", min_value=0.01, value=100.0 if m_market == "US" else 2500.0,
                                   step=1.0, format="%.2f", key="m_S")
        with p2:
            m_K = st.number_input(f"Strike Price ({m_currency})", min_value=0.01, value=100.0 if m_market == "US" else 2500.0,
                                   step=1.0, format="%.2f", key="m_K")
        with p3:
            m_T_days = st.number_input("Days to Expiry", min_value=1, value=30, step=1, key="m_T")
        m_T = m_T_days / 365.0

        p4, p5, p6 = st.columns(3)
        default_r = get_default_rate(m_market)
        with p4:
            m_r_pct = st.number_input("Risk-Free Rate (%)",
                                       min_value=0.0, value=round(default_r * 100, 1),
                                       step=0.1, format="%.1f", key="m_r",
                                       help="India: ~6.8% (10yr G-sec) / US: ~5.3% (10yr Treasury)")
        with p5:
            m_sigma_pct = st.number_input("Volatility σ (%)", min_value=0.1, value=25.0,
                                           step=0.5, format="%.1f", key="m_sigma")
        with p6:
            m_q_pct = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.0,
                                       step=0.1, format="%.2f", key="m_q")

    with right_col:
        st.markdown('<div class="sec-title">Trading Setup</div>', unsafe_allow_html=True)
        m_otype = st.selectbox("Option Type", ["Call", "Put"], key="m_otype")
        m_mkt_price = st.number_input(f"Market Premium ({m_currency})", min_value=0.0,
                                       value=0.0, step=0.1, format="%.2f", key="m_mktprice",
                                       help="Enter observed market price for IV & signal. Leave 0 to skip.")
        m_threshold = st.slider("Signal Threshold (%)", 1.0, 25.0, 5.0, 0.5, key="m_thresh",
                                 help="Min % mispricing to trigger BUY/SELL")

        st.divider()
        st.markdown('<div class="info-banner"><strong>Auto defaults:</strong><br>'
                    'India: r = 6.8% (RBI G-Sec)<br>'
                    'US: r = 5.3% (Fed Treasury)</div>', unsafe_allow_html=True)

    m_r = m_r_pct / 100.0
    m_sigma = m_sigma_pct / 100.0
    m_q = m_q_pct / 100.0

    if st.button("Analyze Option", width='stretch', type="primary", key="m_go"):
        try:
            result = analyze_option(
                S=m_S, K=m_K, T=m_T, r=m_r, sigma=m_sigma, q=m_q,
                market_price=m_mkt_price if m_mkt_price > 0 else None,
                option_type=m_otype.lower(),
                signal_threshold=m_threshold,
                market=m_market,
            )
            curr = result["currency"]

            st.divider()

            # ── Pricing Row ──────────────────────────────────────────
            st.markdown('<div class="sec-title">Theoretical Valuation</div>', unsafe_allow_html=True)
            vc1, vc2, vc3, vc4, vc5 = st.columns(5)
            with vc1:
                st.metric("Call Price", f"{curr}{result['theoretical_price']['call']:,.4f}")
            with vc2:
                st.metric("Put Price", f"{curr}{result['theoretical_price']['put']:,.4f}")
            with vc3:
                iv = result.get("implied_volatility")
                st.metric("Implied Volatility", f"{iv*100:.2f}%" if iv else ("—" if m_mkt_price > 0 else "N/A"))
            with vc4:
                parity = result["put_call_parity"]
                st.metric("Put-Call Parity", "Holds" if parity["holds"] else "Violated",
                          f"err: {parity['error']:.2e}")
            with vc5:
                st.metric("Time to Expiry", f"{m_T_days}d", f"= {m_T:.4f} yrs")

            # ── Signal ───────────────────────────────────────────────
            if "valuation" in result:
                st.divider()
                st.markdown('<div class="sec-title">Trading Signal</div>', unsafe_allow_html=True)
                sig = result["valuation"]
                conf = result.get("confidence", {})
                sc1, sc2, sc3, sc4 = st.columns([1, 1.5, 1.5, 2])
                with sc1:
                    render_signal_badge(sig["signal"], "")
                with sc2:
                    st.metric("Market Price", f"{curr}{result['market_price']:,.2f}")
                with sc3:
                    theo = result["theoretical_price"][m_otype.lower()]
                    st.metric("Theoretical Price", f"{curr}{theo:,.4f}")
                    dev = sig.get("deviation_pct", 0)
                    st.caption(f"Mispricing: {dev:+.2f}%")
                with sc4:
                    if conf:
                        render_confidence(conf)
                st.info(sig["reasoning"])

            # ── Greeks ────────────────────────────────────────────────
            st.divider()
            st.markdown(f'<div class="sec-title">{m_otype} Greeks</div>', unsafe_allow_html=True)
            render_greeks(result["greeks"], m_otype.lower())

            # ── Charts ────────────────────────────────────────────────
            st.divider()
            chart1, chart2 = st.columns(2, gap="medium")
            with chart1:
                premium = result["theoretical_price"][m_otype.lower()]
                st.plotly_chart(plot_payoff(m_S, m_K, premium, m_otype.lower(), curr),
                                width='stretch')
            with chart2:
                st.plotly_chart(
                    plot_greeks_sensitivity(m_S, m_K, m_T, m_r, m_sigma, m_q, m_otype.lower(), curr),
                    width='stretch',
                )

            # ── Formula Reference ─────────────────────────────────────
            with st.expander("Black–Scholes Formulas Reference"):
                st.markdown(f"""
**d₁** = [ ln(S/K) + (r − q + σ²/2)·T ] / (σ·√T)
&nbsp;&nbsp;→ d₁ = **{result['inputs']['spot_price'] and (math.log(m_S/m_K) + (m_r - m_q + 0.5*m_sigma**2)*m_T) / (m_sigma*math.sqrt(m_T)):.4f}**

**d₂** = d₁ − σ·√T

| | Call | Put |
|--|--|--|
| Price | S·e^(-qT)·N(d₁) − K·e^(-rT)·N(d₂) | K·e^(-rT)·N(−d₂) − S·e^(-qT)·N(−d₁) |
| **Δ** | e^(-qT)·N(d₁) | −e^(-qT)·N(−d₁) |
| **Γ** | e^(-qT)·n(d₁)/(S·σ·√T) | same |
| **Θ** | −[S·σ·e^(-qT)·n(d₁)]/(2√T) − rKe^(-rT)N(d₂) | + |
| **ν** | S·e^(-qT)·n(d₁)·√T / 100 | same |
| **ρ** | K·T·e^(-rT)·N(d₂)/100 | −K·T·e^(-rT)·N(−d₂)/100 |

*N(·): Standard normal CDF &nbsp;·&nbsp; n(·): Standard normal PDF*
                """)
        except Exception as e:
            st.error(f"Computation error: {e}")
            import traceback; st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE MARKET
# ═══════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="sec-title">Live Option Chain Analysis</div>', unsafe_allow_html=True)

    lc1, lc2, lc3, lc4 = st.columns([2, 1, 1, 1])
    with lc1:
        live_sym = st.text_input("Ticker Symbol", "RELIANCE.NS", key="l_sym",
                                  help="India: RELIANCE.NS, TCS.NS, INFY.NS | US: AAPL, TSLA, SPY")
    with lc2:
        live_market = detect_market(live_sym)
        default_live_r = get_default_rate(live_market)
        live_r_pct = st.number_input("Risk-Free Rate (%)", value=round(default_live_r * 100, 1),
                                      step=0.1, key="l_r", min_value=0.0)
    with lc3:
        live_expiry_days = st.number_input("Expiry (days, India)", value=28, min_value=1,
                                            step=1, key="l_exp",
                                            help="Used for synthetic chain (Indian stocks)")
    with lc4:
        live_threshold = st.number_input("Signal Threshold (%)", value=5.0, step=0.5,
                                          key="l_thresh", min_value=1.0)

    if st.button("Fetch and Analyze", width='stretch', type="primary", key="l_go"):
        live_market = detect_market(live_sym)
        live_currency = get_currency_symbol(live_market)
        live_r = live_r_pct / 100.0

        render_market_tag(live_market)

        with st.spinner(f"Fetching data for {live_sym}…"):
            spot, hvol, div_yield = fetch_spot_and_vol(live_sym)

            if spot is None:
                st.error(f"Could not fetch price data for **{live_sym}**. Check ticker symbol.")
                st.stop()

            st.success(f"**{live_sym}** · Spot: **{live_currency}{spot:,.2f}** · "
                       f"Hist.Vol: **{hvol*100:.1f}%** · "
                       f"Div Yield: **{div_yield*100:.2f}%**")

        # ── Indian market: synthetic chain ────────────────────────────
        if live_market == "IN":
            st.info("Indian stock options are not available via yfinance. "
                    "Displaying a **BS-priced synthetic option chain** computed from "
                    "live spot price + historical volatility.")

            calls_df, puts_df, expiry_str, is_synth = build_indian_option_chain(
                live_sym, spot, hvol, live_r, int(live_expiry_days)
            )
            T_live = live_expiry_days / 365.0
            data_source = "Synthetic (BS-priced)"

        else:
            # ── US market: real yfinance chain ────────────────────────
            result_us = fetch_us_options(live_sym)
            if len(result_us) == 6:
                calls_raw, puts_raw, _spot, expiry_str, _hvol, div_yield = result_us
            else:
                calls_raw = puts_raw = _spot = expiry_str = _hvol = div_yield = None

            if calls_raw is None:
                st.error("No options data from yfinance. Try a different US ticker (e.g. SPY, AAPL).")
                st.stop()

            exp_date = datetime.strptime(expiry_str, "%Y-%m-%d")
            T_live = max((exp_date - datetime.now()).days / 365.0, 0.001)

            def _prep_chain(df):
                out = []
                for _, row in df.iterrows():
                    K_val = float(row["strike"])
                    mp = float(row.get("lastPrice", 0))
                    iv_raw = float(row.get("impliedVolatility", 0))
                    if mp <= 0.01 or K_val < spot * 0.80 or K_val > spot * 1.20:
                        continue
                    out.append({"strike": K_val, "lastPrice": mp,
                                "impliedVolatility": iv_raw if iv_raw > 0.01 else hvol,
                                "contractSymbol": str(row.get("contractSymbol", ""))})
                return pd.DataFrame(out)

            calls_df = _prep_chain(calls_raw)
            puts_df = _prep_chain(puts_raw)
            data_source = "Live / Delayed (yfinance)"
            is_synth = False

        st.caption(f"Option chain source: **{data_source}** · Expiry: **{expiry_str}** · "
                   f"T = **{T_live:.4f} yrs**")

        # ── Process both sides ────────────────────────────────────────
        def process_chain(df, opt_type):
            rows = []
            for _, row in df.iterrows():
                K_val = float(row["strike"])
                mkt_p = float(row["lastPrice"])
                iv_in = float(row["impliedVolatility"])
                try:
                    res = analyze_option(
                        S=spot, K=K_val, T=T_live, r=live_r,
                        sigma=iv_in, q=div_yield,
                        market_price=mkt_p,
                        option_type=opt_type,
                        signal_threshold=live_threshold,
                        market=live_market,
                    )
                    g = res["greeks"][opt_type]
                    theo = res["theoretical_price"][opt_type]
                    iv_s = res.get("implied_volatility")
                    sig = res.get("valuation", {})
                    conf = res.get("confidence", {})
                    moneyness = "ATM" if abs(K_val - spot) / spot < 0.02 else (
                        "ITM" if (opt_type == "call" and K_val < spot) or
                                 (opt_type == "put" and K_val > spot) else "OTM")
                    rows.append({
                        "Strike": K_val,
                        "Moneyness": moneyness,
                        f"Market {live_currency}": round(mkt_p, 2),
                        f"BS Theoretical {live_currency}": round(theo, 2),
                        "IV Input %": round(iv_in * 100, 1),
                        "IV Solved %": round(iv_s * 100, 2) if iv_s else "—",
                        "Mispricing %": round(sig.get("deviation_pct", 0), 2),
                        "Δ": round(g["delta"], 3),
                        "Γ": round(g["gamma"], 4),
                        "Θ": round(g["theta"], 3),
                        "ν":  round(g["vega"], 3),
                        "ρ":  round(g["rho"], 3),
                        "Signal": sig.get("signal", "—"),
                        "Confidence": f"{conf.get('score', 0):.0f}/100",
                    })
                except Exception:
                    continue
            return pd.DataFrame(rows)

        def style_df(df):
            def _signal_color(val):
                if val == "BUY":   return "background-color:#0d2818;color:#3fb950;font-weight:700"
                if val == "SELL":  return "background-color:#2d1117;color:#f85149;font-weight:700"
                if val == "HOLD":  return "background-color:#2d2100;color:#e3b341;font-weight:700"
                return ""
            def _money_color(val):
                if val == "ATM": return "color:#58a6ff;font-weight:600"
                if val == "ITM": return "color:#3fb950"
                return "color:#6e7681"
            return (df.style
                    .applymap(_signal_color, subset=["Signal"])
                    .applymap(_money_color, subset=["Moneyness"])
                    .format(precision=2))

        call_tab, put_tab = st.tabs([f"Calls ({live_sym})", f"Puts ({live_sym})"])

        for tab_obj, opt_t, df_side in [(call_tab, "call", calls_df), (put_tab, "put", puts_df)]:
            with tab_obj:
                result_df = process_chain(df_side, opt_t)
                if result_df.empty:
                    st.warning(f"No {opt_t} options to display.")
                    continue

                st.dataframe(style_df(result_df), width='stretch', height=420)

                buys  = len(result_df[result_df["Signal"] == "BUY"])
                sells = len(result_df[result_df["Signal"] == "SELL"])
                holds = len(result_df[result_df["Signal"] == "HOLD"])
                s1, s2, s3, s4 = st.columns(4)
                with s1: st.metric("BUY (Undervalued)", buys)
                with s2: st.metric("SELL (Overvalued)", sells)
                with s3: st.metric("HOLD (Fair Value)", holds)
                with s4: st.metric("Total Analyzed", len(result_df))

                # Download
                csv_data = result_df.to_csv(index=False)
                st.download_button(f"Download {opt_t.capitalize()} Analysis CSV",
                                   csv_data, f"{live_sym}_{opt_t}_bs_analysis.csv",
                                   mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH (multi-stock)
# ═══════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sec-title">Batch Analysis — Multiple Stocks</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Enter multiple tickers (one per line). The system will:
    - Fetch live spot price + compute historical volatility
    - Auto-detect India vs US market and apply correct risk-free rate
    - Run BS on ATM call and ATM put for nearest expiry
    - Display a consolidated signal dashboard
    """)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        batch_symbols_raw = st.text_area(
            "Ticker Symbols (one per line)",
            "RELIANCE.NS\nTCS.NS\nINFY.NS\nAAPL\nMSFT\nSPY",
            height=180, key="batch_tickers"
        )
    with col_b:
        batch_days = st.number_input("Days to Expiry (ATM)", value=30, min_value=7, step=1)
        batch_threshold = st.number_input("Signal Threshold (%)", value=5.0, step=0.5, min_value=1.0)

    if st.button("Run Batch Analysis", width='stretch', type="primary", key="batch_go"):
        symbols = [s.strip() for s in batch_symbols_raw.strip().splitlines() if s.strip()]
        if not symbols:
            st.warning("Enter at least one ticker.")
        else:
            progress = st.progress(0, text="Starting batch analysis…")
            batch_rows = []

            for idx, sym in enumerate(symbols):
                progress.progress((idx + 1) / len(symbols), text=f"Analyzing {sym}…")
                market = detect_market(sym)
                currency = get_currency_symbol(market)
                r_def = get_default_rate(market)
                T = batch_days / 365.0

                spot, hvol, div = fetch_spot_and_vol(sym)
                if spot is None:
                    batch_rows.append({"Ticker": sym, "Market": market,
                                       "Error": "Data unavailable"})
                    continue

                try:
                    sigma = hvol or (0.20 if market == "IN" else 0.25)
                    call_res = analyze_option(spot, spot, T, r_def, sigma, div or 0,
                                              option_type="call", market=market)
                    put_res  = analyze_option(spot, spot, T, r_def, sigma, div or 0,
                                              option_type="put", market=market)

                    call_theo = call_res["theoretical_price"]["call"]
                    put_theo  = put_res["theoretical_price"]["put"]
                    cg = call_res["greeks"]["call"]

                    batch_rows.append({
                        "Ticker": sym,
                        "Market": "India" if market == "IN" else "US",
                        "Currency": currency,
                        "Spot Price": f"{currency}{spot:,.2f}",
                        "Hist. Vol %": f"{sigma*100:.1f}%",
                        "ATM Call (BS)": f"{currency}{call_theo:,.2f}",
                        "ATM Put (BS)":  f"{currency}{put_theo:,.2f}",
                        "Delta (Call)":  round(cg["delta"], 4),
                        "Gamma":         round(cg["gamma"], 5),
                        "Theta/day":     round(cg["theta"], 4),
                        "Vega/1%":       round(cg["vega"], 4),
                        "Risk-Free r":   f"{r_def*100:.1f}%",
                        "Days to Exp":   batch_days,
                    })
                except Exception as e:
                    batch_rows.append({"Ticker": sym, "Market": market, "Error": str(e)})

            progress.empty()

            if batch_rows:
                batch_df = pd.DataFrame(batch_rows)
                st.dataframe(batch_df, width='stretch', height=450)

                csv_batch = batch_df.to_csv(index=False)
                st.download_button("Download Batch Results", csv_batch,
                                   "batch_bs_analysis.csv", mime="text/csv",
                                   width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — CSV UPLOAD
# ═══════════════════════════════════════════════════════════════════════
with tab_csv:
    st.markdown('<div class="sec-title">CSV Batch Upload</div>', unsafe_allow_html=True)

    st.info("""
    **Required columns**: `S`, `K`, `T` (years) or `T_days`, `r`, `sigma`
    
    **Optional columns**: `q` (div yield), `market_price`, `option_type` (call/put), `market` (IN/US), `symbol`
    """)

    # Sample CSV download
    sample = pd.DataFrame([
        {"symbol": "RELIANCE.NS", "market": "IN", "S": 2950, "K": 2950, "T_days": 28,
         "r": 0.068, "sigma": 0.22, "q": 0.005, "market_price": 85.0, "option_type": "call"},
        {"symbol": "AAPL", "market": "US", "S": 180, "K": 182.5, "T_days": 21,
         "r": 0.053, "sigma": 0.28, "q": 0.006, "market_price": 8.5, "option_type": "call"},
        {"symbol": "TCS.NS", "market": "IN", "S": 3800, "K": 3750, "T_days": 35,
         "r": 0.068, "sigma": 0.19, "q": 0.0, "market_price": 0, "option_type": "put"},
    ])
    st.download_button("Download Sample CSV Template",
                       sample.to_csv(index=False),
                       "bs_sample_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"], key="csv_up")

    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            if "T_days" in df_in.columns and "T" not in df_in.columns:
                df_in["T"] = df_in["T_days"] / 365.0

            required = ["S", "K", "T", "r", "sigma"]
            missing = [c for c in required if c not in df_in.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.stop()

            st.markdown(f"**{len(df_in)} rows** loaded. Running analysis…")
            prog = st.progress(0)
            results = []

            for idx, row in df_in.iterrows():
                prog.progress((idx + 1) / len(df_in))
                otype = str(row.get("option_type", "call")).lower()
                mkt_code = str(row.get("market", "US")).upper()
                if mkt_code not in ("IN", "US"):
                    sym = str(row.get("symbol", ""))
                    mkt_code = detect_market(sym)
                curr = get_currency_symbol(mkt_code)
                mkt_p = float(row.get("market_price", 0))

                try:
                    res = analyze_option(
                        S=float(row["S"]), K=float(row["K"]),
                        T=float(row["T"]), r=float(row["r"]),
                        sigma=float(row["sigma"]),
                        q=float(row.get("q", 0)),
                        market_price=mkt_p if mkt_p > 0 else None,
                        option_type=otype,
                        market=mkt_code,
                    )
                    g = res["greeks"][otype]
                    entry = {
                        "Symbol": row.get("symbol", "—"),
                        "Market": "IN" if mkt_code == "IN" else "US",
                        "Type": otype.upper(),
                        "Spot": f"{curr}{float(row['S']):,.2f}",
                        "Strike": f"{curr}{float(row['K']):,.2f}",
                        "BS Call": f"{curr}{res['theoretical_price']['call']:,.2f}",
                        "BS Put":  f"{curr}{res['theoretical_price']['put']:,.2f}",
                        "Δ Delta": g["delta"], "Γ Gamma": g["gamma"],
                        "Θ Theta": g["theta"], "ν Vega": g["vega"], "ρ Rho": g["rho"],
                    }
                    if mkt_p > 0:
                        iv = res.get("implied_volatility")
                        sig_r = res.get("valuation", {})
                        conf_r = res.get("confidence", {})
                        entry.update({
                            "Market Price": f"{curr}{mkt_p:,.2f}",
                            "Implied Vol %": f"{iv*100:.2f}%" if iv else "—",
                            "Mispricing %": f"{sig_r.get('deviation_pct', 0):+.1f}%",
                            "Signal": sig_r.get("signal", "—"),
                            "Confidence": f"{conf_r.get('score', 0):.0f}/100",
                        })
                    results.append(entry)
                except Exception as ex:
                    results.append({"Symbol": row.get("symbol", idx), "Error": str(ex)})

            prog.empty()
            df_out = pd.DataFrame(results)

            def _style_csv_signal(val):
                if val == "BUY":  return "background-color:#0d2818;color:#3fb950;font-weight:700"
                if val == "SELL": return "background-color:#2d1117;color:#f85149;font-weight:700"
                return ""

            styled_out = df_out.style
            if "Signal" in df_out.columns:
                styled_out = styled_out.applymap(_style_csv_signal, subset=["Signal"])

            st.dataframe(styled_out, width='stretch', height=500)
            st.download_button("Download Results CSV",
                               df_out.to_csv(index=False),
                               "bs_results.csv", mime="text/csv",
                               width='stretch')

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Quick Reference")
    st.markdown("""
**Indian Tickers**
`RELIANCE.NS` · `TCS.NS` · `INFY.NS`
`HDFCBANK.NS` · `ICICIBANK.NS`

**US Tickers**
`SPY` · `AAPL` · `MSFT` · `TSLA` · `QQQ`

---

**Default Risk-Free Rates**
| Market | Rate | Benchmark |
|--------|------|-----------|
| India | 6.8% | RBI 10yr G-Sec |
| US | 5.3% | Fed 10yr Treasury |

---

**Signal Logic**
| Signal | Condition |
|--------|-----------|
| BUY | Market < BS Theo − threshold |
| SELL | Market > BS Theo + threshold |
| HOLD | Within threshold |

---

**Confidence Score**
- 0–39: Low
- 40–69: Medium
- 70–100: High

Based on: time-to-expiry, moneyness, mispricing magnitude.

---
    """)
    st.caption("Model: Black–Scholes–Merton (1973)")
    st.caption("IV Solver: Brent's method (scipy)")
    st.caption("Data: yfinance (live/delayed)")
    st.caption("For educational use only.")
