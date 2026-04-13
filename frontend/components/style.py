import streamlit as st
import warnings

# Suppress Plotly 6.x DeprecationWarnings — Streamlit 1.50 captures these
# and renders them as visible UI boxes if not suppressed at module level.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")
warnings.filterwarnings("ignore", message=".*keyword arguments.*deprecated.*")
warnings.filterwarnings("ignore", message=".*config.*Plotly.*")
warnings.filterwarnings("ignore", message=".*deprecated.*future release.*")

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Roboto+Mono:wght@300;400;500&display=swap');

/* ── Base ─────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]  { background: #131722; }
[data-testid="stHeader"]            { background: #131722; border-bottom: 1px solid #2A2E39; }
[data-testid="stToolbar"]           { display: none !important; }
[data-testid="stHeader"] button     { display: none !important; }
header [data-testid="stToolbar"]    { display: none !important; }
.stDeployButton                     { display: none !important; }
[data-testid="manage-app-button"]   { display: none !important; }
button[kind="header"]               { display: none !important; }
section[data-testid="stSidebar"]    { background: #1E222D; border-right: 1px solid #2A2E39; }
section[data-testid="stSidebar"] .stMarkdown p {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.72rem;
    color: #787B86;
}

/* ── Body text ───────────────────────────────────────── */
.main p, .main li {
    font-family: 'Inter', sans-serif;
    color: #787B86;
    font-size: 0.85rem;
    line-height: 1.6;
}
code {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.78rem;
    background: rgba(33,150,243,0.08);
    color: #2196F3;
    padding: 0.1em 0.35em;
    border-radius: 2px;
}
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #D1D4DC !important;
    letter-spacing: -0.01em;
}

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #1E222D;
    border: 1px solid #2A2E39;
    border-radius: 2px;
    padding: 0.875rem 1.125rem;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 0.57rem !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #787B86 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: #D1D4DC;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #2A2E39;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #787B86;
    background: transparent;
    border-bottom: 2px solid transparent;
    padding: 0.6rem 1.25rem;
    margin-bottom: -1px;
    transition: color 0.15s;
}
.stTabs [aria-selected="true"] { color: #2196F3; border-bottom-color: #2196F3; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    border-radius: 2px;
    border: 1px solid #2A2E39;
    color: #D1D4DC;
    background: #1E222D;
    transition: all 0.12s;
}
.stButton > button:hover {
    border-color: #2196F3;
    color: #2196F3;
    background: rgba(33,150,243,0.06);
}
.stButton > button[kind="primary"] {
    border-color: #2196F3;
    color: #131722;
    background: #2196F3;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background: #1976D2;
    border-color: #1976D2;
}

/* ── Text inputs ─────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    background: #1E222D;
    border: 1px solid #2A2E39;
    border-radius: 2px;
    color: #D1D4DC;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.82rem;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2196F3 !important;
    box-shadow: 0 0 0 1px rgba(33,150,243,0.15);
}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stSlider label, .stDateInput label, .stNumberInput label {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 0.57rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #787B86 !important;
}

/* ── Selectbox ───────────────────────────────────────── */
[data-baseweb="select"] > div:first-child {
    background: #1E222D;
    border-color: #2A2E39;
    color: #D1D4DC;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.78rem;
    border-radius: 2px;
}

/* ── Slider ──────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #2196F3;
    border-color: #2196F3;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] {
    color: #787B86;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.6rem;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: #2A2E39; margin: 1.25rem 0; }

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] summary {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #787B86;
    background: #1E222D;
    border: 1px solid #2A2E39;
    border-radius: 2px;
    padding: 0.6rem 0.875rem;
}
[data-testid="stExpander"] > div {
    border-color: #2A2E39;
    background: transparent;
}

/* ── Alerts ──────────────────────────────────────────── */
.stAlert {
    border-radius: 2px;
    background: #1E222D !important;
    border: 1px solid #2A2E39 !important;
}
.stAlert p {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #D1D4DC;
}
[data-testid="stNotification"] { border-left-width: 3px !important; }

/* ── Success / Info / Warning colors ────────────────── */
.stSuccess { border-left-color: #26A69A !important; }
.stInfo    { border-left-color: #2196F3 !important; }
.stWarning { border-left-color: #F7A600 !important; }
.stError   { border-left-color: #EF5350 !important; }

/* ── Progress bar ────────────────────────────────────── */
[data-testid="stProgress"] > div > div { background: #2196F3; }

/* ── Chat ────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: #1E222D;
    border: 1px solid #2A2E39;
    color: #D1D4DC;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.82rem;
}
[data-testid="stChatMessageContent"] p {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #2A2E39;
    border-radius: 2px;
}
[data-testid="stDataFrame"] table {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.78rem;
}
[data-testid="stDataFrame"] th {
    background: #1E222D !important;
    color: #787B86 !important;
    font-size: 0.57rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #2A2E39 !important;
    font-weight: 400;
}
[data-testid="stDataFrame"] td {
    color: #D1D4DC;
    border-bottom: 1px solid #2A2E39 !important;
    background: transparent !important;
}

/* ── Spinner ─────────────────────────────────────────── */
.stSpinner > div { border-top-color: #2196F3 !important; }

/* ── Sidebar nav items ───────────────────────────────── */
section[data-testid="stSidebar"] a {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #787B86;
}
section[data-testid="stSidebar"] a:hover { color: #D1D4DC; }
"""


def inject_css():
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def page_header(label: str, title: str, accent: str = "#2196F3"):
    st.markdown(f"""
    <div style="border-bottom:1px solid #2A2E39;padding-bottom:0.875rem;margin-bottom:1.5rem;">
        <span style="font-family:'Roboto Mono',monospace;font-size:0.57rem;letter-spacing:0.18em;
        color:{accent};text-transform:uppercase;">{label}</span>
        <h1 style="font-family:'Inter',sans-serif;font-size:1.35rem;font-weight:500;
        color:#D1D4DC;margin:0.25rem 0 0 0;letter-spacing:-0.02em;">{title}</h1>
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str, color: str = "#787B86"):
    st.markdown(
        f'<div style="font-family:Roboto Mono,monospace;font-size:0.57rem;letter-spacing:0.15em;'
        f'text-transform:uppercase;color:{color};margin:1rem 0 0.4rem 0;border-left:2px solid {color};padding-left:0.5rem;">{text}</div>',
        unsafe_allow_html=True,
    )
