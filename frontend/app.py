import streamlit as st
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from components.style import inject_css

API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

st.markdown("""
<div style="padding:2.5rem 0 2rem 0;border-bottom:1px solid #1E2D3D;margin-bottom:2.5rem;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.25em;
    color:#7A9BB5;text-transform:uppercase;margin-bottom:0.6rem;">
        Stock Price Predictor
    </div>
    <h1 style="font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:300;
    color:#E8F4FD;margin:0 0 0.8rem 0;letter-spacing:-0.02em;">
        Stock Price Predictor
    </h1>
    <p style="font-family:'IBM Plex Sans',sans-serif;font-size:0.88rem;color:#7A9BB5;
    margin:0;max-width:560px;line-height:1.75;">
        Institutional-grade portfolio analytics combining quantitative optimisation,
        reinforcement learning, and SABR options pricing.
    </p>
</div>
""", unsafe_allow_html=True)

modules = [
    ("#00D4FF", "01 — Dashboard",
     "Intraday trend prediction with OHLC charts, RSI signals, peer comparison, and AI-driven market commentary."),
    ("#00FF94", "02 — Advisor",
     "Conversational AI for investment research, portfolio strategy, and real-time market questions."),
    ("#FFB800", "03 — Portfolio",
     "Multi-strategy allocation via Riskfolio-Lib and RL ensemble (A2C · SAC · TD3) with blending controls."),
    ("#BF7FFF", "04 — Options",
     "SABR-calibrated mispricing detection with live NSE/global option chains and 3D volatility surface."),
]

cols = st.columns(4, gap="medium")
for col, (color, title, desc) in zip(cols, modules):
    with col:
        st.markdown(f"""
        <div style="border:1px solid #1E2D3D;border-top:2px solid {color};
        padding:1.25rem;border-radius:2px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.88rem;font-weight:500;
            color:#E8F4FD;margin-bottom:0.55rem;">{title}</div>
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.75rem;
            color:#7A9BB5;line-height:1.65;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.15em;
color:#2A3D50;text-transform:uppercase;margin-top:2.5rem;">
    Select a module from the sidebar to get started.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.2em;
color:#7A9BB5;text-transform:uppercase;padding:0.25rem 0;">Modules</div>
""", unsafe_allow_html=True)
