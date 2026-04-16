import streamlit as st
import os, sys
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from components.style import inject_css

API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="QuantVision",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .module-card {
      border: 1px solid #1E2D3D;
      border-radius: 12px;
      padding: 1.5rem;
      background: #0A0E14;
      height: 100%;
  }
  .module-card h4 { color: #00D4FF; margin-bottom: 0.5rem; }
  .module-card p  { color: #7A9BB5; font-size: 0.9rem; line-height: 1.5; }
  .tag {
      display: inline-block;
      background: #1E2D3D;
      color: #00D4FF;
      border-radius: 4px;
      padding: 2px 8px;
      font-size: 0.75rem;
      margin-right: 4px;
      margin-top: 6px;
  }
  .hero-label {
      letter-spacing: 0.2em;
      font-size: 0.75rem;
      color: #7A9BB5;
      text-transform: uppercase;
      margin-top: 1rem;
      display: block;
  }
  .divider { border-top: 1px solid #1E2D3D; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# --- Hero ---------------------------------------------------------------
st.title("Institutional-Grade Market Analytics")
st.markdown(
    "A full-stack quantitative research platform combining deep learning, "
    "classical econometrics, and reinforcement learning for rigorous "
    "price prediction and portfolio optimization."
)

# --- Module cards -------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('''
    <div class="module-card">
      <h4>Dashboard</h4>
      <p>Bidirectional LSTM price forecasting with 7 technical features (RSI, MACD, Bollinger Bands, ATR, SMA distance, OBV). Weekly resampling reduces noise. 30-day dampened momentum projection with T+1 execution simulation.</p>
      <span class="tag">LSTM</span>
      <span class="tag">BiLSTM</span>
      <span class="tag">Technical Analysis</span>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div class="module-card">
      <h4>Portfolio Optimizer</h4>
      <p>Three-profile allocation engine: inverse-volatility (conservative), minimum-variance (balanced), and maximum-Sharpe (aggressive). Blended via risk tolerance slider. Black-Litterman with LSTM-derived views. RL ensemble (A2C + SAC + TD3) for dynamic rebalancing.</p>
      <span class="tag">Black-Litterman</span>
      <span class="tag">RL Ensemble</span>
      <span class="tag">PyPortfolioOpt</span>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    <div class="module-card">
      <h4>Options Analysis</h4>
      <p>Black-Scholes pricing engine with full Greeks. SABR stochastic volatility model for IV surface calibration. Mispricing detection at 2% threshold. 3D volatility mesh across strikes and expiries.</p>
      <span class="tag">Black-Scholes</span>
      <span class="tag">SABR</span>
      <span class="tag">Greeks</span>
      <span class="tag">IV Surface</span>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown('''
    <div class="module-card">
      <h4>Backtest Engine</h4>
      <p>Point-in-time validation with strict train/test separation. LSTM versus ARIMA+GARCH comparison on identical historical windows. T+1 open execution with 0.1% fees and 0.05% slippage. Equity curve, Sharpe ratio, max drawdown, and per-row price verification log.</p>
      <span class="tag">ARIMA</span>
      <span class="tag">GARCH</span>
      <span class="tag">Zero Lookahead</span>
      <span class="tag">Walk-Forward</span>
    </div>
    ''', unsafe_allow_html=True)

# --- Methodology --------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("## How It Works")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("### Data Pipeline")
    st.markdown(
        "Market data is sourced via yfinance with Polygon.io as primary. "
        "Raw OHLCV is resampled to weekly frequency to reduce microstructure "
        "noise. Seven features are computed: RSI(14), MACD(12/26/9), "
        "Bollinger Band width, ATR(14), SMA(20) distance, OBV, and closing "
        "price. All features are MinMax-scaled to [0,1] before model input."
    )

with m2:
    st.markdown("### Prediction Engine")
    st.markdown(
        "A Bidirectional LSTM processes sequences of 20 weekly windows. "
        "The model predicts log returns (stationary target) rather than "
        "price levels, eliminating non-stationarity. Predictions are "
        "inverse-transformed via price(T) = price(T-1) × exp(log_return). "
        "Hyperparameters are evolved via Genetic Algorithm and cached with "
        "a 3-day TTL."
    )

with m3:
    st.markdown("### Validation Framework")
    st.markdown(
        "All backtests enforce strict temporal separation. The model is "
        "instantiated and trained only on data before the selected cutoff "
        "date. ARIMA+GARCH provides a classical statistical baseline for "
        "comparison. Metrics include RMSE, MAE, directional accuracy, "
        "Sharpe ratio, max drawdown, and a per-prediction verification log."
    )

# --- System status ------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("## System Status")
try:
    r = requests.get(f"{API_URL}/hpo-status/?symbol=AAPL", timeout=3)
    if r.status_code == 200:
        st.success("Backend API is online and responding.")
    else:
        st.warning("Backend API returned an unexpected response.")
except:
    st.error("Backend API is offline. Start the Django server to use this platform.")
