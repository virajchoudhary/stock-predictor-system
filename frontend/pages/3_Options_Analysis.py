import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
import requests

# ---------------------------------------------------------------------------
# NSE Live Data — jugaad-data (replaces nsepython scraper)
# nsepython used to scrape NSE's website and gets blocked / breaks constantly.
# jugaad-data uses NSE's official data endpoints and is actively maintained.
# ---------------------------------------------------------------------------
try:
    from jugaad_data.nse import NSELive
    _nse = NSELive()
    JUGAAD_AVAILABLE = True
except ImportError:
    JUGAAD_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Options Analysis - QuantVis",
    layout="wide",
)

st.title("Options Mispricing Detector (SABR Model)")

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_option_chain(symbol):
    """
    Fetch option chain data for a given symbol.
    Priority order:
      1. jugaad-data  (NIFTY / BANKNIFTY — live NSE data)
      2. yfinance     (US stocks and ETFs — live/delayed data)
      3. Simulation   (fallback for NSE when live fetch fails)
    """

    # ------------------------------------------------------------------
    # 1. jugaad-data for NIFTY / BANKNIFTY
    # ------------------------------------------------------------------
    nifty_symbols = ["^NSEI", "NIFTY", "NIFTY 50", "NIFTY50"]
    banknifty_symbols = ["^NSEBANK", "BANKNIFTY", "BANK NIFTY"]

    if JUGAAD_AVAILABLE and symbol.upper() in [s.upper() for s in nifty_symbols + banknifty_symbols]:
        index_name = "BANKNIFTY" if symbol.upper() in [s.upper() for s in banknifty_symbols] else "NIFTY"
        try:
            payload   = _nse.live_option_chain(index_name)
            records   = payload.get("records", {})
            spot_price = records.get("underlyingValue", 24000.0)
            expiry_list = records.get("expiryDates", [])

            if not expiry_list:
                raise ValueError("No expiry dates returned from NSE.")

            nearest_expiry = expiry_list[0]
            data_list = []

            for item in records.get("data", []):
                if item.get("expiryDate") == nearest_expiry and "CE" in item:
                    ce = item["CE"]
                    iv_raw = ce.get("impliedVolatility", 0)
                    data_list.append({
                        "strike":            ce["strikePrice"],
                        "lastPrice":         ce["lastPrice"],
                        # NSE returns IV as percentage (e.g. 15.5 means 15.5%)
                        "impliedVolatility": iv_raw / 100.0 if iv_raw > 0 else 0.0,
                        "contractSymbol":    ce.get("identifier", f"CE_{ce['strikePrice']}"),
                    })

            if data_list:
                return pd.DataFrame(data_list), float(spot_price), nearest_expiry

        except Exception as e:
            st.warning(
                f"jugaad-data live fetch failed for {index_name}: `{e}`. "
                "Falling back to simulation mode."
            )
            # Fall through to simulation below

    # ------------------------------------------------------------------
    # 2. yfinance for US / global symbols
    # ------------------------------------------------------------------
    ticker = yf.Ticker(symbol)
    try:
        expirations = ticker.options

        if not expirations:
            # No options data — use simulation for Indian indices
            if any(x in symbol.upper() for x in ["NSE", ".NS", "NIFTY", "NSEI"]):
                return _simulate_nifty_chain(symbol)
            return None, None, None

        expiry = expirations[0]
        chain  = ticker.option_chain(expiry)

        hist          = ticker.history(period="1d")
        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 100.0

        return chain.calls, current_price, expiry

    except Exception:
        # Last-resort simulation for NSE
        if any(x in symbol.upper() for x in ["NSE", ".NS", "NIFTY", "NSEI"]):
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

tab_mispricing, tab_bs, tab_surface = st.tabs([
    "Mispricing (SABR)",
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
            help="US: SPY, AAPL, QQQ  |  India: NIFTY or NIFTY 50 (uses jugaad-data live NSE feed)",
        )
        if JUGAAD_AVAILABLE:
            st.caption("jugaad-data installed — NIFTY live data available.")
        else:
            st.caption(
                "jugaad-data not installed. "
                "Run `pip install jugaad-data` for live NSE options. "
                "NIFTY will use simulation mode."
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
                st.plotly_chart(fig_smile, use_container_width=True)

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
                    use_container_width=True,
                )

            else:
                st.warning("Not enough liquid options data to calibrate SABR.")
        else:
            st.error(
                "Could not fetch option chain. "
                "Ticker might be invalid or no options data is available."
            )

# ---------------------------------------------------------------------------
# TAB 2 — Black-Scholes
# ---------------------------------------------------------------------------
with tab_bs:
    from scipy.stats import norm as sp_norm

    def bs_price(S, K, T, r, sigma, option_type="call"):
        """Black-Scholes option price."""
        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return intrinsic
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S * sp_norm.cdf(d1) - K * np.exp(-r * T) * sp_norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * sp_norm.cdf(-d2) - S * sp_norm.cdf(-d1)

    def bs_greeks(S, K, T, r, sigma, option_type="call"):
        """Compute all Greeks."""
        if T <= 0 or sigma <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = sp_norm.pdf(d1)

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * pdf_d1 * np.sqrt(T) / 100  # per 1% move

        if option_type == "call":
            delta = sp_norm.cdf(d1)
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * sp_norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * sp_norm.cdf(d2) / 100
        else:
            delta = sp_norm.cdf(d1) - 1
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * sp_norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * sp_norm.cdf(-d2) / 100

        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

    def bs_implied_vol(market_price, S, K, T, r, option_type="call"):
        """Newton's method for implied volatility."""
        sigma = 0.2
        for _ in range(100):
            price = bs_price(S, K, T, r, sigma, option_type)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega_raw = S * sp_norm.pdf(d1) * np.sqrt(T)
            if vega_raw < 1e-10:
                break
            sigma = sigma - (price - market_price) / vega_raw
            sigma = max(sigma, 0.001)
        return sigma

    st.subheader("Black-Scholes Option Pricer")

    # Inputs
    bs_col1, bs_col2 = st.columns(2)
    with bs_col1:
        bs_S = st.number_input("Spot Price (S)", 1.0, 100000.0, 100.0, 1.0, key="bs_S")
        bs_K = st.number_input("Strike Price (K)", 1.0, 100000.0, 100.0, 1.0, key="bs_K")
        bs_T = st.number_input("Time to Expiry (years)", 0.01, 10.0, 1.0, 0.01, key="bs_T")
    with bs_col2:
        bs_r = st.number_input("Risk-Free Rate", 0.0, 0.30, 0.05, 0.005, key="bs_r")
        bs_sigma = st.number_input("Volatility (sigma)", 0.01, 3.0, 0.20, 0.01, key="bs_sigma")
        bs_type = st.selectbox("Option Type", ["call", "put"], key="bs_type")

    # Price and Greeks
    price_call = bs_price(bs_S, bs_K, bs_T, bs_r, bs_sigma, "call")
    price_put = bs_price(bs_S, bs_K, bs_T, bs_r, bs_sigma, "put")
    greeks = bs_greeks(bs_S, bs_K, bs_T, bs_r, bs_sigma, bs_type)

    st.divider()

    # Metrics row
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Call Price", f"${price_call:.4f}")
    with mc2:
        st.metric("Put Price", f"${price_put:.4f}")
    with mc3:
        st.metric("Put-Call Parity Check",
                   f"${price_call - price_put:.4f}",
                   delta=f"S - K*e^(-rT) = ${bs_S - bs_K * np.exp(-bs_r * bs_T):.4f}")
    with mc4:
        st.metric("Intrinsic Value",
                   f"${max(bs_S - bs_K, 0) if bs_type == 'call' else max(bs_K - bs_S, 0):.4f}")

    # Greeks
    st.subheader("Greeks")
    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    with gc1:
        st.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
    with gc2:
        st.metric("Gamma (Γ)", f"{greeks['gamma']:.6f}")
    with gc3:
        st.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
    with gc4:
        st.metric("Vega (ν)", f"{greeks['vega']:.4f}")
    with gc5:
        st.metric("Rho (ρ)", f"{greeks['rho']:.4f}")

    st.divider()

    # --- Charts ---
    st.subheader("Payoff & Sensitivity Charts")
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Payoff Diagram", "Greeks vs Spot", "Price Heatmap"])

    with chart_tab1:
        spot_range = np.linspace(bs_K * 0.5, bs_K * 1.5, 200)
        payoff_call = np.maximum(spot_range - bs_K, 0) - price_call
        payoff_put = np.maximum(bs_K - spot_range, 0) - price_put
        price_curve = np.array([
            bs_price(s, bs_K, bs_T, bs_r, bs_sigma, bs_type) for s in spot_range
        ])
        intrinsic = np.maximum(spot_range - bs_K, 0) if bs_type == "call" else np.maximum(bs_K - spot_range, 0)

        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(
            x=spot_range, y=price_curve,
            mode="lines", name=f"BS {bs_type.title()} Price",
            line=dict(color="#00CC96", width=2),
        ))
        fig_payoff.add_trace(go.Scatter(
            x=spot_range, y=intrinsic,
            mode="lines", name="Intrinsic Value",
            line=dict(color="#636EFA", dash="dash"),
        ))
        fig_payoff.add_trace(go.Scatter(
            x=spot_range,
            y=payoff_call if bs_type == "call" else payoff_put,
            mode="lines", name="P&L at Expiry",
            line=dict(color="#EF553B", width=1.5),
        ))
        fig_payoff.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_payoff.add_vline(x=bs_K, line_dash="dot", line_color="gray",
                             annotation_text="Strike")
        fig_payoff.update_layout(
            xaxis_title="Spot Price",
            yaxis_title="Value ($)",
            margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

    with chart_tab2:
        spot_g = np.linspace(bs_K * 0.5, bs_K * 1.5, 200)
        deltas = [bs_greeks(s, bs_K, bs_T, bs_r, bs_sigma, bs_type)["delta"] for s in spot_g]
        gammas = [bs_greeks(s, bs_K, bs_T, bs_r, bs_sigma, bs_type)["gamma"] for s in spot_g]
        thetas = [bs_greeks(s, bs_K, bs_T, bs_r, bs_sigma, bs_type)["theta"] for s in spot_g]
        vegas = [bs_greeks(s, bs_K, bs_T, bs_r, bs_sigma, bs_type)["vega"] for s in spot_g]

        from plotly.subplots import make_subplots
        fig_greeks = make_subplots(rows=2, cols=2,
                                   subplot_titles=["Delta", "Gamma", "Theta", "Vega"])
        fig_greeks.add_trace(go.Scatter(x=spot_g, y=deltas, line=dict(color="#636EFA")), row=1, col=1)
        fig_greeks.add_trace(go.Scatter(x=spot_g, y=gammas, line=dict(color="#00CC96")), row=1, col=2)
        fig_greeks.add_trace(go.Scatter(x=spot_g, y=thetas, line=dict(color="#EF553B")), row=2, col=1)
        fig_greeks.add_trace(go.Scatter(x=spot_g, y=vegas, line=dict(color="#AB63FA")), row=2, col=2)
        fig_greeks.update_layout(
            showlegend=False,
            margin=dict(t=40, b=30),
            height=500,
        )
        for i in range(1, 5):
            fig_greeks.update_xaxes(title_text="Spot Price", row=(i-1)//2+1, col=(i-1)%2+1)
        st.plotly_chart(fig_greeks, use_container_width=True)

    with chart_tab3:
        st.caption("Option price as a function of Spot Price and Volatility.")
        spot_hm = np.linspace(bs_K * 0.7, bs_K * 1.3, 30)
        vol_hm = np.linspace(0.05, 0.80, 25)
        price_grid = np.array([
            [bs_price(s, bs_K, bs_T, bs_r, v, bs_type) for s in spot_hm]
            for v in vol_hm
        ])
        fig_hm = go.Figure(data=go.Heatmap(
            z=price_grid,
            x=np.round(spot_hm, 1),
            y=np.round(vol_hm, 2),
            colorscale="Viridis",
            colorbar_title="Price ($)",
        ))
        fig_hm.update_layout(
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # --- Implied Volatility Calculator ---
    st.divider()
    st.subheader("Implied Volatility Calculator")
    iv_col1, iv_col2 = st.columns([2, 1])
    with iv_col1:
        bs_market_price = st.number_input(
            "Market Option Price", 0.01, 100000.0, 10.0, 0.01, key="bs_mkt_price"
        )
    with iv_col2:
        if st.button("Calculate IV", key="bs_calc_iv"):
            iv = bs_implied_vol(bs_market_price, bs_S, bs_K, bs_T, bs_r, bs_type)
            st.metric("Implied Volatility", f"{iv:.4f} ({iv*100:.2f}%)")

# ---------------------------------------------------------------------------
# TAB 3 — Volatility Surface
# ---------------------------------------------------------------------------
with tab_surface:
    st.subheader("3D Implied Volatility Surface")
    st.info(
        "Visualising the volatility structure across Strikes and Time. "
        "(Mock data generated for visualisation — real surface requires fetching multiple expiries.)"
    )

    spot_ref    = spot_price if "spot_price" in dir() and spot_price else 150.0
    strikes_vis = np.linspace(spot_ref * 0.8, spot_ref * 1.2, 20)
    times_vis   = np.linspace(0.1, 1.0, 10)

    S_mesh, T_mesh = np.meshgrid(strikes_vis, times_vis)
    IV_mesh = 0.2 + 0.1 * ((S_mesh - spot_ref) / spot_ref) ** 2 + 0.05 * np.exp(-T_mesh)

    fig_surf = go.Figure(data=[go.Surface(z=IV_mesh, x=strikes_vis, y=times_vis)])
    fig_surf.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Expiry (Years)",
            zaxis_title="Implied Volatility",
        ),
    )
    st.plotly_chart(fig_surf, use_container_width=True)


