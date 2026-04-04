import streamlit as st

st.set_page_config(
    page_title="QuantVis",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080C10;
    color: #E8F4FD;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #1E2D3D;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Hide default Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Buttons */
.stButton > button {
    background-color: #FF4444;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.2rem;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #FF6666;
}

/* Inputs */
.stTextInput > div > input,
.stTextArea > div > textarea,
.stSelectbox > div > div {
    background-color: #111820;
    border: 1px solid #1E2D3D;
    color: #E8F4FD;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}

/* Dataframe */
.stDataFrame { border: 1px solid #1E2D3D; }

/* Metric */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: #00D4FF;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #00D4FF;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #3A5570;
}
.stTabs [aria-selected="true"] {
    color: #00D4FF;
    border-bottom: 2px solid #FF4444;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar logo / branding ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 2rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;
                    font-weight:500;color:#E8F4FD;letter-spacing:0.05em;">
            ▲ QuantVis
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                    color:#3A5570;letter-spacing:0.2em;text-transform:uppercase;
                    margin-top:0.2rem;">
            Quantitative Vision
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Landing page content ──────────────────────────────────────────────────────
st.markdown("""
<div style="max-width:700px;margin:4rem auto 0 auto;text-align:center;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                letter-spacing:0.3em;color:#00D4FF;text-transform:uppercase;
                margin-bottom:1rem;">
        QUANTITATIVE VISION
    </div>
    <h1 style="font-family:'IBM Plex Mono',monospace;font-size:2.2rem;
               font-weight:500;color:#E8F4FD;margin-bottom:1rem;line-height:1.3;">
        AI-Powered<br>Portfolio Intelligence
    </h1>
    <p style="font-family:'IBM Plex Sans',sans-serif;font-size:0.95rem;
              color:#7A9BB5;line-height:1.8;margin-bottom:2.5rem;">
        Combines classical quant optimisation with FinRL reinforcement learning,
        LSTM price prediction with evolved hyperparameters, and AI-scored
        news sentiment — all in one platform.
    </p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3, col4 = st.columns(4)

cards = [
    ("📊", "Dashboard", "Live price predictions with LSTM + technical indicators", "Dashboard"),
    ("💬", "Chat",      "AI financial assistant powered by Groq LLaMA",           "Chat"),
    ("▲",  "Portfolio", "FinRL ensemble allocator blended with Riskfolio-Lib",    "Portfolio"),
    ("⚡", "Options",   "Options analysis and strategy visualisation",             "Options Analysis"),
]

for col, (icon, title, desc, _) in zip([col1, col2, col3, col4], cards):
    with col:
        st.markdown(f"""
        <div style="background:#0D1117;border:1px solid #1E2D3D;
                    border-top:2px solid #00D4FF;padding:1.2rem;
                    height:160px;position:relative;">
            <div style="font-size:1.4rem;margin-bottom:0.5rem;">{icon}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                        color:#E8F4FD;font-weight:500;margin-bottom:0.4rem;">{title}</div>
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.72rem;
                        color:#3A5570;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:2rem;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                 color:#3A5570;letter-spacing:0.15em;">
        SELECT A PAGE FROM THE SIDEBAR TO GET STARTED
    </span>
</div>
""", unsafe_allow_html=True)
