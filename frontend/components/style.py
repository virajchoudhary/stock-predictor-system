import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;1,300&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ── Base ─────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]  { background: #080C10; }
[data-testid="stHeader"]            { background: transparent; border-bottom: 1px solid #1E2D3D; }
[data-testid="stToolbar"]           { display: none; }
section[data-testid="stSidebar"]    { background: #0A0E14; border-right: 1px solid #1E2D3D; }
section[data-testid="stSidebar"] .stMarkdown p { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #7A9BB5; }

/* ── Body text ───────────────────────────────────────── */
.main p, .main li { font-family: 'IBM Plex Sans', sans-serif; color: #7A9BB5; font-size: 0.85rem; line-height: 1.7; }
code { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; background: rgba(0,212,255,0.07); color: #00D4FF; padding: 0.1em 0.35em; border-radius: 2px; }

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0A0E14;
    border: 1px solid #1E2D3D;
    border-radius: 3px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.58rem !important;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #7A9BB5 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: #E8F4FD;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1E2D3D;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7A9BB5;
    background: transparent;
    border-bottom: 2px solid transparent;
    padding: 0.65rem 1.5rem;
    margin-bottom: -1px;
    transition: color 0.15s;
}
.stTabs [aria-selected="true"] { color: #00D4FF; border-bottom-color: #00D4FF; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 2px;
    border: 1px solid #1E2D3D;
    color: #7A9BB5;
    background: transparent;
    transition: all 0.15s;
}
.stButton > button:hover  { border-color: #00D4FF; color: #00D4FF; background: rgba(0,212,255,0.05); }
.stButton > button[kind="primary"]       { border-color: #00D4FF; color: #00D4FF; background: rgba(0,212,255,0.07); }
.stButton > button[kind="primary"]:hover { background: rgba(0,212,255,0.14); }

/* ── Text inputs ─────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    background: #0D1117;
    border: 1px solid #1E2D3D;
    border-radius: 2px;
    color: #E8F4FD;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.15);
}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stSlider label, .stDateInput label, .stNumberInput label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7A9BB5 !important;
}

/* ── Selectbox ───────────────────────────────────────── */
[data-baseweb="select"] > div:first-child {
    background: #0D1117;
    border-color: #1E2D3D;
    color: #E8F4FD;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    border-radius: 2px;
}

/* ── Divider / separator ─────────────────────────────── */
hr { border-color: #1E2D3D; margin: 1.5rem 0; }

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7A9BB5;
}
[data-testid="stExpander"] > div {
    border-color: #1E2D3D;
    background: transparent;
}

/* ── Alerts ──────────────────────────────────────────── */
.stAlert { border-radius: 2px; }
.stAlert p { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.82rem; }

/* ── Progress bar ────────────────────────────────────── */
[data-testid="stProgress"] > div > div { background: #00D4FF; }

/* ── Chat ────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: #0D1117;
    border: 1px solid #1E2D3D;
    color: #E8F4FD;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
[data-testid="stChatMessageContent"] p { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.85rem; }

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #1E2D3D; border-radius: 2px; }

/* ── Spinner ─────────────────────────────────────────── */
.stSpinner > div { border-top-color: #00D4FF !important; }
"""


def inject_css():
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def page_header(label: str, title: str, accent: str = "#00D4FF"):
    st.markdown(f"""
    <div style="border-bottom:1px solid #1E2D3D;padding-bottom:1rem;margin-bottom:1.75rem;">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.22em;
        color:{accent};text-transform:uppercase;">{label}</span>
        <h1 style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:400;
        color:#E8F4FD;margin:0.3rem 0 0 0;letter-spacing:-0.01em;">{title}</h1>
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str, color: str = "#7A9BB5"):
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;letter-spacing:0.18em;'
        f'text-transform:uppercase;color:{color};margin:1rem 0 0.5rem 0;">▸ {text}</div>',
        unsafe_allow_html=True,
    )
