import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*keyword arguments.*deprecated.*")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
import requests
import time
import uuid
import os
import sys

_FRONTEND_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if _FRONTEND_ROOT not in sys.path:
    sys.path.insert(0, _FRONTEND_ROOT)

from components.black_scholes_ui import render_black_scholes_analyzer
from components.sidebar import render_backtest_sidebar

API_URL = "http://127.0.0.1:8000/api"

# ---------------------------------------------------------------------------
# Option-chain sourcing
# Use yfinance where options are available and fall back to a simulated
# NIFTY-style chain for symbols that do not expose live option chains.
# ---------------------------------------------------------------------------
nifty_symbols = {"^NSEI", "NIFTY", "NIFTY 50", "NIFTY50"}
banknifty_symbols = {"^NSEBANK", "BANKNIFTY", "BANK NIFTY"}


def _uses_simulated_nse_chain(symbol):
    upper_symbol = symbol.upper()
    return (
        upper_symbol in {s.upper() for s in nifty_symbols | banknifty_symbols}
        or any(token in upper_symbol for token in ["NSE", ".NS", ".BO", "NIFTY"])
    )

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Options Analysis - Stock Price Predictor",
    layout="wide",
)
target_date = render_backtest_sidebar()

st.title("Options Mispricing Detector (SABR Model)")

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_option_chain(symbol):
    """
    Fetch option chain data for a given symbol.
    Priority order:
      1. yfinance   (US stocks and ETFs — live/delayed data)
      2. Simulation (fallback for NSE / unavailable live chains)
    """

    # ------------------------------------------------------------------
    # 1. yfinance for symbols with listed options
    # ------------------------------------------------------------------
    ticker = yf.Ticker(symbol)
    try:
        expirations = ticker.options

        if not expirations:
            if _uses_simulated_nse_chain(symbol):
                return _simulate_nifty_chain(symbol)
            return None, None, None

        expiry = expirations[0]
        chain  = ticker.option_chain(expiry)

        hist          = ticker.history(period="1d")
        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 100.0

        return chain.calls, current_price, expiry

    except Exception:
        if _uses_simulated_nse_chain(symbol):
            return _simulate_nifty_chain(symbol)
        return None, None, None


def _simulate_nifty_chain(symbol):
    """
    Generate a realistic simulated Nifty option chain for SABR demo purposes
    when live data is unavailable. Clearly labelled as simulated in the UI.
    """
    current_price = 24500.0
    expiry        = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    strikes       = np.arange(current_price * 0.95, current_price * 1.05, 50)

    # Realistic volatility smile: ATM IV + skew + curvature
    atm_iv  = 0.15
    log_m   = np.log(strikes / current_price)
    mkt_ivs = atm_iv - 0.5 * log_m + 2.0 * log_m ** 2
    mkt_price = 100.0 * np.exp(-log_m)

    calls = pd.DataFrame({
        "strike":            strikes,
        "lastPrice":         mkt_price,
        "impliedVolatility": mkt_ivs,
        "contractSymbol":    [f"Simulated_Call_{int(s)}" for s in strikes],
    })
    return calls, current_price, expiry


def calculate_time_to_expiry(expiry_str):
    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
    delta       = expiry_date - datetime.now()
    return max(delta.days / 365.0, 0.001)


# ---------------------------------------------------------------------------
# SABR Model
# ---------------------------------------------------------------------------

def sabr_vol(k, f, t, alpha, beta, rho, nu):
    """
    SABR Volatility (Hagan et al. 2002).
    k: strike, f: forward (spot proxy), t: time to expiry
    alpha, beta, rho, nu: SABR parameters
    """
    try:
        if k <= 0 or f <= 0 or t <= 0:
            return 0.0

        log_fk   = np.log(f / k)
        fk_beta  = (f * k) ** ((1 - beta) / 2)
        z        = (nu / alpha) * fk_beta * log_fk

        if abs(z) < 1e-5:
            x_z = 1.0
        else:
            arg = 1 - 2 * rho * z + z * z
            if arg < 0:
                arg = 0.0
            x_z = np.log((np.sqrt(arg) + z - rho) / (1 - rho)) / z

        term1    = (1 - beta) ** 2 / 24 * log_fk ** 2
        term2    = (rho * beta * nu * alpha) / (4 * fk_beta)
        term3    = (2 - 3 * rho ** 2) * nu ** 2 / 24
        brackets = 1 + (term1 + term2 + term3) * t

        vol = (alpha / fk_beta) * (z / x_z if abs(x_z) > 1e-5 else 1.0) * brackets
        return float(vol)
    except Exception:
        return 0.0


def calibrate_sabr(strikes, market_ivs, f, t):
    """Calibrate SABR parameters (alpha, rho, nu) to market IVs. Beta fixed at 0.5."""
    beta = 0.5

    def objective(params):
        alpha, rho, nu = params
        sabr_ivs = np.array([sabr_vol(k, f, t, alpha, beta, rho, nu) for k in strikes])
        return np.sum((sabr_ivs - market_ivs) ** 2)

    atm_vol       = float(np.mean(market_ivs))
    initial_guess = [atm_vol, 0.0, 0.5]
    bounds        = [(0.01, 2.0), (-0.99, 0.99), (0.01, 5.0)]

    result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")
    return result.x  # alpha, rho, nu


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

tab_mispricing, tab_rl_blend, tab_bs, tab_surface = st.tabs([
    "Mispricing (SABR)",
    "RL Ensemble Blend",
    "Black-Scholes",
    "Volatility Surface",
])

# ---------------------------------------------------------------------------
# TAB 1 — SABR Mispricing
# ---------------------------------------------------------------------------
with tab_mispricing:
    col_input, col_status = st.columns([1, 2])

    with col_input:
        symbol = st.text_input(
            "Index / Stock Symbol",
            "SPY",
            help="US: SPY, AAPL, QQQ  |  India: NIFTY or NIFTY 50 (falls back to a simulated NSE chain when live options are unavailable)",
        )
        st.caption(
            "US options use yfinance live/delayed chains. "
            "NIFTY-style symbols fall back to a simulated chain when live options are unavailable."
        )

    if symbol:
        calls_df, spot_price, expiry_date = fetch_option_chain(symbol)

        if calls_df is not None and spot_price:
            is_simulated = "Simulated" in str(calls_df["contractSymbol"].iloc[0])

            if is_simulated:
                st.warning(
                    f"**SIMULATION MODE** — Live data unavailable for `{symbol}`. "
                    "Displaying a realistic Nifty volatility smile to demonstrate SABR calibration."
                )
            else:
                st.success(f"**LIVE / DELAYED DATA** — Option chain fetched for `{symbol}`.")

            with col_status:
                st.metric("Spot Price", f"₹{spot_price:,.2f}", f"Expiry: {expiry_date}")

            # Filter near-the-money strikes
            upper = spot_price * 1.10
            lower = spot_price * 0.90
            liquid_calls = calls_df[
                (calls_df["strike"] > lower) & (calls_df["strike"] < upper)
            ].copy()

            liquid_calls["strike"]            = pd.to_numeric(liquid_calls["strike"],            errors="coerce")
            liquid_calls["impliedVolatility"] = pd.to_numeric(liquid_calls["impliedVolatility"], errors="coerce")
            liquid_calls.dropna(subset=["strike", "impliedVolatility", "lastPrice"], inplace=True)
            liquid_calls = liquid_calls[liquid_calls["impliedVolatility"] > 0.05]

            if not liquid_calls.empty:
                t = calculate_time_to_expiry(expiry_date)

                strikes    = liquid_calls["strike"].values
                market_ivs = liquid_calls["impliedVolatility"].values

                try:
                    valid_mask = np.isfinite(market_ivs) & (market_ivs > 0)
                    if np.sum(valid_mask) < 3:
                        st.warning("Not enough valid data points for calibration (need > 3).")
                        st.stop()

                    strikes    = strikes[valid_mask]
                    market_ivs = market_ivs[valid_mask]

                    alpha, rho, nu = calibrate_sabr(strikes, market_ivs, spot_price, t)

                    liquid_calls["SABR_IV"] = [
                        sabr_vol(k, spot_price, t, alpha, 0.5, rho, nu)
                        for k in liquid_calls["strike"]
                    ]
                    liquid_calls["Mispricing"] = liquid_calls["impliedVolatility"] - liquid_calls["SABR_IV"]

                except Exception as e:
                    st.error(f"Calibration Failed: {e}")
                    st.stop()

                def classify(row):
                    threshold = 0.02
                    if row["Mispricing"] >  threshold: return "Overpriced (SELL)"
                    if row["Mispricing"] < -threshold: return "Undervalued (BUY)"
                    return "Fair Value"

                liquid_calls["Signal"] = liquid_calls.apply(classify, axis=1)

                st.subheader("SABR Calibration Result")
                st.write(
                    f"**Alpha**: {alpha:.4f} | **Beta**: 0.5 (Fixed) | "
                    f"**Rho**: {rho:.4f} | **Nu**: {nu:.4f}"
                )

                fig_smile = go.Figure()
                fig_smile.add_trace(go.Scatter(
                    x=liquid_calls["strike"], y=liquid_calls["impliedVolatility"],
                    mode="markers", name="Market IV",
                ))
                fig_smile.add_trace(go.Scatter(
                    x=liquid_calls["strike"], y=liquid_calls["SABR_IV"],
                    mode="lines", name="SABR Fair IV",
                    line=dict(color="orange"),
                ))
                fig_smile.update_layout(
                    title=f"Volatility Smile: {symbol} ({expiry_date})",
                    xaxis_title="Strike Price",
                    yaxis_title="Implied Volatility",
                )
                st.plotly_chart(fig_smile, width='stretch')

                st.subheader("Mispricing Alerts")
                st.dataframe(
                    liquid_calls[[
                        "strike", "lastPrice", "impliedVolatility",
                        "SABR_IV", "Mispricing", "Signal",
                    ]].style.applymap(
                        lambda x: (
                            "color: red"   if "SELL" in str(x) else
                            "color: green" if "BUY"  in str(x) else ""
                        ),
                        subset=["Signal"],
                    ),
                    width='stretch',
                )

            else:
                st.warning("Not enough liquid options data to calibrate SABR.")
        else:
            st.error(
                "Could not fetch option chain. "
                "Ticker might be invalid or no options data is available."
            )

# ---------------------------------------------------------------------------
# TAB 2 — RL Ensemble Blend (A2C · SAC · TD3)
# ---------------------------------------------------------------------------
with tab_rl_blend:
    st.markdown("### RL Ensemble Portfolio Blend")
    st.caption("Train A2C · SAC · TD3 agents on your option underlyings and blend their allocations into a final weight.")

    SPEED_MAP = {"Fast (5k)": 5000, "Medium (20k)": 20000, "Thorough (50k)": 50000}
    COLORS    = ["#2196F3", "#26A69A", "#F7A600", "#EF5350", "#9C27B0", "#FF6D00"]

    def _donut(alloc, centre):
        fig = go.Figure(data=[go.Pie(
            labels=list(alloc.keys()), values=list(alloc.values()), hole=0.55,
            marker=dict(colors=COLORS[:len(alloc)], line=dict(color="#131722", width=2)),
            insidetextfont=dict(size=10, color="#D1D4DC"),
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            annotations=[dict(text=centre, x=0.5, y=0.5, font=dict(size=13, color="#D1D4DC"), showarrow=False)],
        )
        return fig

    LABELS = {
        "fetching_data":          "Fetching 2 years of price data...",
        "fetching_fundamentals":  "Scoring fundamentals & news sentiment...",
        "hpo_running":            "Running GA hyperparameter optimisation...",
        "hpo_complete":           "HPO complete — evolved hyperparameters ready.",
        "hpo_loaded_from_cache":  "HPO loaded from cache.",
        "hpo_skipped":            "HPO skipped — using defaults.",
        "training_A2C":           "[1/3] Training A2C...",
        "training_SAC":           "[2/3] Training SAC...",
        "training_TD3":           "[3/3] Training TD3...",
        "evaluating":             "Computing ensemble allocation...",
        "complete":               "Done.",
    }

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_t, col_c = st.columns([3, 2])
    with col_t:
        rl_tickers_input = st.text_area("UNDERLYING TICKERS", "AAPL, MSFT, GOOG", height=70,
                                        key="rl_opt_tickers")
    with col_c:
        rl_speed   = st.selectbox("RL TRAINING SPEED", list(SPEED_MAP.keys()), index=0, key="rl_opt_speed")
        rl_ts      = SPEED_MAP[rl_speed]
        hpo_eligible = rl_ts >= 10000
        use_hpo    = st.toggle("Evolve Hyperparameters (GA HPO)", value=True,
                               disabled=not hpo_eligible,
                               help="Genetic Algorithm evolves optimal hyperparameters. Requires Medium or Thorough speed.",
                               key="rl_opt_hpo")
        if not hpo_eligible:
            st.caption("HPO requires Medium or Thorough speed.")
        elif use_hpo:
            st.caption("GA will evolve hyperparameters for A2C · SAC · TD3.")
        else:
            st.caption("Default hyperparameters will be used.")

    rl_blend = st.slider("QUANT ← BLEND → RL ENSEMBLE", 0.0, 1.0, 0.5, step=0.05, key="rl_opt_blend")
    quant_pct = round((1 - rl_blend) * 100)
    rl_pct    = round(rl_blend * 100)
    st.caption(f"{quant_pct}% Quant (Min-Vol)  +  {rl_pct}% RL Ensemble (A2C · SAC · TD3)")

    for k, v in [("rl_alloc_quant", None), ("rl_alloc_rl", None),
                 ("rl_alloc_blend", None), ("rl_session", str(uuid.uuid4())[:8]),
                 ("rl_training_attempted", False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    run_rl = st.button("Train and Blend", type="primary", key="rl_opt_run")

    if run_rl:
        tickers = [t.strip() for t in rl_tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.warning("Enter at least 2 tickers.")
        else:
            st.session_state.rl_alloc_quant = None
            st.session_state.rl_alloc_rl    = None
            st.session_state.rl_alloc_blend = None
            st.session_state.rl_session     = str(uuid.uuid4())[:8]

            # Step 1 — Quant (min-vol via optimize endpoint)
            with st.spinner("Step 1 — Running Quant optimisation..."):
                try:
                    resp = requests.post(f"{API_URL}/optimize/",
                                         json={"tickers": tickers, "risk_tolerance": 0.3, "target_date": target_date},
                                         timeout=30)
                    if resp.status_code == 200:
                        q_raw = resp.json().get("allocation", {})
                        st.session_state.rl_alloc_quant = q_raw
                    else:
                        st.warning("Quant step failed — RL-only blend will be used.")
                except Exception as e:
                    st.warning(f"Quant error: {e}")

            # Step 2 — RL Ensemble (async train + poll)
            if rl_blend > 0:
                st.session_state.rl_training_attempted = True
                st.markdown("**Step 2 — Training RL Ensemble (A2C · SAC · TD3)**")
                pb   = st.progress(0)
                stxt = st.empty()
                try:
                    kick = requests.post(f"{API_URL}/rl/train/", json={
                        "tickers":    tickers,
                        "timesteps":  rl_ts,
                        "session_id": st.session_state.rl_session,
                        "use_hpo":    use_hpo,
                        "target_date": target_date,
                    }, timeout=10)
                    if kick.status_code == 200:
                        for _ in range(600):
                            try:
                                prog = requests.get(
                                    f"{API_URL}/rl/progress/{st.session_state.rl_session}/",
                                    timeout=5).json()
                                pct    = prog.get("progress", 0)
                                status = prog.get("status", "")
                                pb.progress(int(min(pct, 100)))
                                stxt.caption(f"{LABELS.get(status, status)}  {pct:.0f}%")
                                if status == "error":
                                    st.error(prog.get("error", "Unknown error"))
                                    break
                                if status == "complete":
                                    st.session_state.rl_alloc_rl = prog.get("allocation", {})
                                    break
                            except Exception:
                                pass
                            time.sleep(2)
                    else:
                        st.warning("RL training failed to start.")
                except Exception as e:
                    st.warning(f"RL error: {e}")

            # Blend quant + RL
            q = st.session_state.rl_alloc_quant or {}
            r = st.session_state.rl_alloc_rl    or {}
            if q or r:
                if q and r:
                    all_t  = set(q) | set(r)
                    blended = {t: (1 - rl_blend) * q.get(t, 0.0) + rl_blend * r.get(t, 0.0) for t in all_t}
                    total  = sum(blended.values())
                    blended = {t: round(v / total, 4) for t, v in blended.items() if v > 0.001}
                else:
                    blended = q or r
                st.session_state.rl_alloc_blend = blended
            st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.rl_alloc_blend:
        q     = st.session_state.rl_alloc_quant or {}
        r     = st.session_state.rl_alloc_rl    or {}
        blend = st.session_state.rl_alloc_blend

        st.success("BLENDED ALLOCATION READY")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption(f"Quant (Min-Vol) — {quant_pct}%")
            if q: st.plotly_chart(_donut(q, "Quant"), width='stretch')
        with c2:
            st.caption(f"RL Ensemble — {rl_pct}%")
            if r:
                st.plotly_chart(_donut(r, "ENS"), width='stretch')
            elif rl_pct == 0:
                st.info("RL weight set to 0% — increase the blend slider to include RL.")
            else:
                st.info(f"RL Ensemble ({rl_pct}%) — training not yet complete. Click **Train and Blend** to run.")
        with c3:
            st.caption(f"Final Blend — {quant_pct}% + {rl_pct}%")
            st.plotly_chart(_donut(blend, "MIX"), width='stretch')

        # Comparison table
        all_t = sorted(set(list(q.keys()) + list(r.keys()) + list(blend.keys())))
        rows  = []
        for t in all_t:
            qw = q.get(t, 0.0); rw = r.get(t, 0.0); bw = blend.get(t, 0.0)
            rows.append({
                "Ticker":       t,
                "Quant (Min-Vol)": f"{qw:.1%}" if qw > 0 else "—",
                "RL Ensemble":  f"{rw:.1%}" if rw > 0 else "—",
                "Final Blend":  f"{bw:.1%}" if bw > 0 else "—",
                "Δ RL vs Quant": (f"+{bw-qw:.1%}" if bw-qw >= 0 else f"{bw-qw:.1%}") if qw > 0 else "new",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        # Agent metrics
        prog_data = {}
        try:
            prog_data = requests.get(
                f"{API_URL}/rl/progress/{st.session_state.rl_session}/", timeout=3).json()
        except Exception:
            pass
        if prog_data.get("agent_metrics"):
            st.markdown("**Agent Performance**")
            m_rows = []
            for agent, m in prog_data["agent_metrics"].items():
                m_rows.append({
                    "Agent":        agent,
                    "Final Value":  f"${m['final_value']:,.0f}",
                    "Return":       f"{m['total_return']:+.1f}%",
                    "Sharpe":       f"{m['sharpe']:.3f}",
                    "Sortino":      f"{m['sortino']:.3f}",
                    "Max Drawdown": f"{m['max_drawdown']:.2%}",
                })
            st.dataframe(pd.DataFrame(m_rows), width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# TAB 3 — Black-Scholes
# ---------------------------------------------------------------------------
with tab_bs:
    render_black_scholes_analyzer()

# ---------------------------------------------------------------------------
# TAB 4 — Volatility Surface
# ---------------------------------------------------------------------------
with tab_surface:
    st.subheader("3D Implied Volatility Surface")

    vol_symbol = st.text_input(
        "Symbol for IV Surface",
        "SPY",
        key="vol_surface_symbol",
        help="Enter a US stock/ETF with listed options (e.g. SPY, AAPL, QQQ, MSFT).",
    )

    @st.cache_data(ttl=600, show_spinner="Fetching option chains across expiries...")
    def _build_iv_surface(sym):
        """Fetch real IV data from multiple expiries via yfinance."""
        ticker = yf.Ticker(sym)
        try:
            expirations = ticker.options
        except Exception:
            return None, None, None, None

        if not expirations or len(expirations) < 2:
            return None, None, None, None

        hist = ticker.history(period="1d")
        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if current_price is None:
            return None, None, None, None

        all_strikes = []
        all_ttes = []
        all_ivs = []

        now = datetime.now()
        # Use up to 8 expiries for a good surface
        for exp_str in expirations[:8]:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                tte = max((exp_date - now).days / 365.0, 0.01)

                chain = ticker.option_chain(exp_str)
                calls = chain.calls.copy()
                calls["strike"] = pd.to_numeric(calls["strike"], errors="coerce")
                calls["impliedVolatility"] = pd.to_numeric(calls["impliedVolatility"], errors="coerce")
                calls = calls.dropna(subset=["strike", "impliedVolatility"])
                calls = calls[calls["impliedVolatility"] > 0.01]

                # Filter to strikes within ±25% of spot
                lower = current_price * 0.75
                upper = current_price * 1.25
                calls = calls[(calls["strike"] >= lower) & (calls["strike"] <= upper)]

                for _, row in calls.iterrows():
                    all_strikes.append(float(row["strike"]))
                    all_ttes.append(tte)
                    all_ivs.append(float(row["impliedVolatility"]))

            except Exception:
                continue

        if len(all_strikes) < 10:
            return None, None, None, current_price

        return (
            np.array(all_strikes),
            np.array(all_ttes),
            np.array(all_ivs),
            current_price,
        )

    if vol_symbol:
        strikes_arr, ttes_arr, ivs_arr, spot_ref_val = _build_iv_surface(vol_symbol.strip().upper())

        if strikes_arr is not None and len(strikes_arr) >= 10:
            st.success(
                f"**Live IV Surface** — Built from {len(strikes_arr)} option contracts "
                f"across multiple expiries for `{vol_symbol.strip().upper()}` (spot: ${spot_ref_val:,.2f})."
            )

            fig_surf = go.Figure(data=[go.Mesh3d(
                x=strikes_arr,
                y=ttes_arr,
                z=ivs_arr,
                intensity=ivs_arr,
                colorscale="Viridis",
                opacity=0.85,
                colorbar=dict(title="IV"),
                hovertemplate=(
                    "Strike: $%{x:,.0f}<br>"
                    "Time to Expiry: %{y:.2f}y<br>"
                    "Implied Vol: %{z:.1%}<extra></extra>"
                ),
            )])
            fig_surf.update_layout(
                title=f"Implied Volatility Surface — {vol_symbol.strip().upper()}",
                scene=dict(
                    xaxis_title="Strike ($)",
                    yaxis_title="Time to Expiry (Years)",
                    zaxis_title="Implied Volatility",
                ),
                height=600,
            )
            st.plotly_chart(fig_surf, width='stretch')

            st.caption(
                "This surface is constructed from real call-option implied volatilities "
                "fetched via yfinance across multiple listed expiration dates. "
                "It reveals the volatility smile/skew and term structure of the underlying."
            )
        else:
            # Parametric fallback
            fallback_spot = spot_ref_val if spot_ref_val else 150.0
            st.warning(
                f"Live option chain data unavailable for `{vol_symbol.strip().upper()}`. "
                "Displaying a parametric volatility surface model."
            )
            strikes_vis = np.linspace(fallback_spot * 0.8, fallback_spot * 1.2, 20)
            times_vis   = np.linspace(0.1, 1.0, 10)
            S_mesh, T_mesh = np.meshgrid(strikes_vis, times_vis)
            IV_mesh = 0.2 + 0.1 * ((S_mesh - fallback_spot) / fallback_spot) ** 2 + 0.05 * np.exp(-T_mesh)

            fig_surf = go.Figure(data=[go.Surface(z=IV_mesh, x=strikes_vis, y=times_vis, colorscale="Viridis")])
            fig_surf.update_layout(
                title="Parametric Implied Volatility Surface (Model)",
                scene=dict(
                    xaxis_title="Strike",
                    yaxis_title="Time to Expiry (Years)",
                    zaxis_title="Implied Volatility",
                ),
                height=600,
            )
            st.plotly_chart(fig_surf, width='stretch')
            st.caption(
                "This is a parametric model surface (not live data). "
                "Enter a US equity ticker with listed options (e.g. SPY, AAPL) to see real implied volatilities."
            )
