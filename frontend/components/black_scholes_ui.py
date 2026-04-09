import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_BACKEND_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "backend")
)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from api.black_scholes import (  # noqa: E402
    analyze_option,
    bs_price,
    confidence_score,
    detect_market,
    get_currency_symbol,
    get_default_rate,
    greeks,
    historical_volatility,
)


def _inject_css():
    st.markdown(
        """
        <style>
        .bs-note {
            border: 1px solid #1E2D3D;
            border-radius: 6px;
            padding: 12px 14px;
            background: rgba(10, 14, 20, 0.9);
            margin-bottom: 16px;
        }
        .bs-signal-buy { color: #4CAF50; font-weight: 600; }
        .bs-signal-sell { color: #EF553B; font-weight: 600; }
        .bs-signal-hold { color: #FECB52; font-weight: 600; }
        .bs-greek-card {
            border: 1px solid #1E2D3D;
            border-radius: 6px;
            padding: 12px;
            background: rgba(10, 14, 20, 0.9);
            text-align: center;
        }
        .bs-greek-label {
            color: #7A9BB5;
            font-size: 0.72rem;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .bs-greek-value {
            color: #E8F4FD;
            font-size: 1.05rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _signal_class(signal):
    signal = str(signal).upper()
    if signal == "BUY":
        return "bs-signal-buy"
    if signal == "SELL":
        return "bs-signal-sell"
    return "bs-signal-hold"


def _signal_style(signal):
    signal = str(signal).upper()
    if signal == "BUY":
        return "color: #4CAF50; font-weight: 600"
    if signal == "SELL":
        return "color: #EF553B; font-weight: 600"
    return "color: #FECB52; font-weight: 600"


def _render_signal(signal_payload):
    if not signal_payload:
        return
    signal = signal_payload.get("signal", "HOLD")
    reasoning = signal_payload.get("reasoning", "")
    st.markdown(
        f'<div class="bs-note"><div class="{_signal_class(signal)}">{signal}</div><div>{reasoning}</div></div>',
        unsafe_allow_html=True,
    )


def _render_greeks(greek_payload, option_type):
    selected = greek_payload.get(option_type.lower(), {})
    cols = st.columns(5)
    labels = [
        ("Delta", selected.get("delta", 0.0)),
        ("Gamma", selected.get("gamma", 0.0)),
        ("Theta / day", selected.get("theta", 0.0)),
        ("Vega / 1%", selected.get("vega", 0.0)),
        ("Rho / 1%", selected.get("rho", 0.0)),
    ]
    for col, (label, value) in zip(cols, labels):
        with col:
            st.markdown(
                f"""
                <div class="bs-greek-card">
                    <div class="bs-greek-label">{label}</div>
                    <div class="bs-greek-value">{value:+.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _plot_payoff(spot, strike, premium, time_to_expiry, rate, sigma, dividend_yield, option_type, currency):
    price_range = np.linspace(max(spot * 0.5, 1.0), spot * 1.5, 250)
    if option_type == "call":
        expiry_pnl = np.maximum(price_range - strike, 0.0) - premium
    else:
        expiry_pnl = np.maximum(strike - price_range, 0.0) - premium

    theoretical_curve = np.array([
        bs_price(px, strike, time_to_expiry, rate, sigma, q=dividend_yield, option_type=option_type)
        for px in price_range
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_range,
        y=theoretical_curve,
        mode="lines",
        name="Theoretical Price",
        line=dict(color="#00CC96", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=price_range,
        y=expiry_pnl,
        mode="lines",
        name="Expiry PnL",
        line=dict(color="#EF553B", width=2),
    ))
    fig.add_vline(x=strike, line_dash="dot", line_color="#636EFA")
    fig.add_hline(y=0, line_dash="dash", line_color="#7A9BB5")
    fig.update_layout(
        title=f"{option_type.title()} payoff and theoretical value",
        xaxis_title="Spot Price",
        yaxis_title=f"Value ({currency})",
        margin=dict(t=40, b=30),
    )
    return fig


def _plot_greek_sensitivity(spot, strike, time_to_expiry, rate, sigma, dividend_yield, option_type):
    spot_range = np.linspace(max(spot * 0.6, 1.0), spot * 1.4, 160)
    greek_curves = {"Delta": [], "Gamma": [], "Theta": [], "Vega": []}

    for px in spot_range:
        greek_data = greeks(px, strike, time_to_expiry, rate, sigma, dividend_yield)[option_type]
        greek_curves["Delta"].append(greek_data["delta"])
        greek_curves["Gamma"].append(greek_data["gamma"])
        greek_curves["Theta"].append(greek_data["theta"])
        greek_curves["Vega"].append(greek_data["vega"])

    fig = make_subplots(rows=2, cols=2, subplot_titles=list(greek_curves.keys()))
    colors = {
        "Delta": "#636EFA",
        "Gamma": "#00CC96",
        "Theta": "#EF553B",
        "Vega": "#AB63FA",
    }
    positions = {"Delta": (1, 1), "Gamma": (1, 2), "Theta": (2, 1), "Vega": (2, 2)}
    for label, values in greek_curves.items():
        row, col = positions[label]
        fig.add_trace(
            go.Scatter(x=spot_range, y=values, mode="lines", line=dict(color=colors[label], width=2)),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Spot Price", row=row, col=col)

    fig.update_layout(height=460, margin=dict(t=50, b=30), showlegend=False)
    return fig


def _plot_price_heatmap(strike, time_to_expiry, rate, option_type):
    spot_range = np.linspace(strike * 0.7, strike * 1.3, 30)
    vol_range = np.linspace(0.05, 0.8, 24)
    price_grid = np.array(
        [
            [bs_price(px, strike, time_to_expiry, rate, vol, option_type=option_type) for px in spot_range]
            for vol in vol_range
        ]
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=price_grid,
            x=np.round(spot_range, 2),
            y=np.round(vol_range, 2),
            colorscale="Viridis",
            colorbar_title="Price",
        )
    )
    fig.update_layout(
        title="Price heatmap by spot and volatility",
        xaxis_title="Spot Price",
        yaxis_title="Volatility",
        margin=dict(t=40, b=30),
    )
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_spot_and_vol(symbol):
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    history = ticker.history(period="3mo")
    if history is None or history.empty:
        return None, None, None

    close_prices = history["Close"].dropna()
    if close_prices.empty:
        return None, None, None

    info = {}
    try:
        info = ticker.info
    except Exception:
        info = {}

    return (
        float(close_prices.iloc[-1]),
        historical_volatility(close_prices.tolist(), 30),
        float(info.get("dividendYield") or 0.0),
    )


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_us_options(symbol):
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    expiries = ticker.options
    if not expiries:
        return None, None, None, None, None, None

    history = ticker.history(period="3mo")
    if history is None or history.empty:
        return None, None, None, None, None, None

    spot = float(history["Close"].iloc[-1])
    hist_vol = historical_volatility(history["Close"].tolist(), 30)
    expiry = expiries[0]
    option_chain = ticker.option_chain(expiry)
    info = {}
    try:
        info = ticker.info
    except Exception:
        info = {}

    return (
        option_chain.calls.copy(),
        option_chain.puts.copy(),
        spot,
        expiry,
        hist_vol,
        float(info.get("dividendYield") or 0.0),
    )


@st.cache_data(ttl=300, show_spinner=False)
def _build_indian_option_chain(symbol, spot, hist_vol, rate, expiry_days):
    time_to_expiry = expiry_days / 365.0
    expiry = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
    strike_step = 50 if spot > 500 else 5
    strike_levels = np.linspace(spot * 0.88, spot * 1.12, 11)
    strikes = sorted({round(level / strike_step) * strike_step for level in strike_levels})

    calls = []
    puts = []
    for strike in strikes:
        log_moneyness = math.log(max(spot, 1.0) / max(strike, 1.0))
        call_vol = max(hist_vol * (1 + 0.05 * log_moneyness), 0.05)
        put_vol = max(hist_vol * (1 - 0.08 * log_moneyness), 0.05)
        call_price = bs_price(spot, strike, time_to_expiry, rate, call_vol, option_type="call")
        put_price = bs_price(spot, strike, time_to_expiry, rate, put_vol, option_type="put")

        calls.append({
            "strike": strike,
            "lastPrice": round(call_price, 2),
            "impliedVolatility": round(call_vol, 4),
            "contractSymbol": f"{symbol}_CALL_{int(strike)}",
        })
        puts.append({
            "strike": strike,
            "lastPrice": round(put_price, 2),
            "impliedVolatility": round(put_vol, 4),
            "contractSymbol": f"{symbol}_PUT_{int(strike)}",
        })

    return pd.DataFrame(calls), pd.DataFrame(puts), expiry


def _process_option_chain(side_df, option_type, spot, time_to_expiry, rate, dividend_yield, threshold, market_code):
    rows = []
    for _, row in side_df.iterrows():
        strike = float(row.get("strike", 0.0))
        market_price = float(row.get("lastPrice", 0.0))
        sigma = float(row.get("impliedVolatility", 0.0) or 0.0)
        if strike <= 0 or market_price <= 0 or sigma <= 0:
            continue

        result = analyze_option(
            S=spot,
            K=strike,
            T=time_to_expiry,
            r=rate,
            sigma=sigma,
            q=dividend_yield,
            market_price=market_price,
            option_type=option_type,
            signal_threshold=threshold,
            market=market_code,
        )
        valuation = result.get("valuation", {})
        greek_data = result.get("greeks", {}).get(option_type, {})
        rows.append({
            "Strike": round(strike, 2),
            "Market Price": round(market_price, 2),
            "Theoretical Price": result["theoretical_price"][option_type],
            "Input IV %": round(sigma * 100, 2),
            "Solved IV %": (
                round(float(result.get("implied_volatility")) * 100.0, 2)
                if result.get("implied_volatility") is not None
                else None
            ),
            "Mispricing %": valuation.get("deviation_pct"),
            "Signal": valuation.get("signal"),
            "Delta": round(greek_data.get("delta", 0.0), 4),
            "Gamma": round(greek_data.get("gamma", 0.0), 4),
            "Theta": round(greek_data.get("theta", 0.0), 4),
            "Vega": round(greek_data.get("vega", 0.0), 4),
            "Rho": round(greek_data.get("rho", 0.0), 4),
            "Confidence": result.get("confidence", {}).get("score"),
        })
    return pd.DataFrame(rows)


def _render_manual_analysis():
    st.markdown(
        '<div class="bs-note">Manual Black-Scholes-Merton pricing with dividend yield, parity checks, Greeks, implied volatility, and signal confidence.</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        market_choice = st.selectbox(
            "Market",
            ["Auto-Detect", "India", "US"],
            key="bs_manual_market_choice",
        )
        ticker_label = st.text_input("Ticker label", "AAPL", key="bs_manual_ticker")
        if market_choice == "Auto-Detect":
            market_code = detect_market(ticker_label)
        else:
            market_code = "IN" if market_choice == "India" else "US"
        currency = get_currency_symbol(market_code)

        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            spot = st.number_input(
                f"Spot Price ({currency})",
                min_value=0.01,
                value=100.0 if market_code == "US" else 2500.0,
                step=1.0,
                key="bs_manual_spot",
            )
        with s_col2:
            strike = st.number_input(
                f"Strike Price ({currency})",
                min_value=0.01,
                value=100.0 if market_code == "US" else 2500.0,
                step=1.0,
                key="bs_manual_strike",
            )
        with s_col3:
            days_to_expiry = st.number_input(
                "Days to Expiry",
                min_value=1,
                value=30,
                step=1,
                key="bs_manual_days",
            )

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            rate_pct = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                value=round(get_default_rate(market_code) * 100.0, 2),
                step=0.1,
                key="bs_manual_rate",
            )
        with p_col2:
            sigma_pct = st.number_input(
                "Volatility (%)",
                min_value=0.1,
                value=25.0,
                step=0.5,
                key="bs_manual_sigma",
            )
        with p_col3:
            dividend_pct = st.number_input(
                "Dividend Yield (%)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key="bs_manual_dividend",
            )

    with col_right:
        option_type = st.selectbox("Option Type", ["call", "put"], key="bs_manual_type")
        market_price = st.number_input(
            f"Observed Market Price ({currency})",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="bs_manual_market_price",
        )
        threshold_pct = st.slider(
            "Signal Threshold (%)",
            min_value=1.0,
            max_value=25.0,
            value=5.0,
            step=0.5,
            key="bs_manual_threshold",
        )
        st.caption(
            f"Default rate for {market_code}: {get_default_rate(market_code) * 100.0:.2f}%."
        )

    time_to_expiry = days_to_expiry / 365.0
    result = analyze_option(
        S=spot,
        K=strike,
        T=time_to_expiry,
        r=rate_pct / 100.0,
        sigma=sigma_pct / 100.0,
        q=dividend_pct / 100.0,
        market_price=market_price if market_price > 0 else None,
        option_type=option_type,
        signal_threshold=threshold_pct,
        market=market_code,
    )

    metrics = st.columns(4)
    with metrics[0]:
        st.metric("Call Price", f"{currency}{result['theoretical_price']['call']:.4f}")
    with metrics[1]:
        st.metric("Put Price", f"{currency}{result['theoretical_price']['put']:.4f}")
    with metrics[2]:
        st.metric(
            "Parity Error",
            f"{result['put_call_parity']['error']:.6f}",
            delta="Holds" if result["put_call_parity"]["holds"] else "Check inputs",
        )
    with metrics[3]:
        selected_price = result["theoretical_price"][option_type]
        st.metric(f"{option_type.title()} Value", f"{currency}{selected_price:.4f}")

    if result.get("valuation"):
        _render_signal(result["valuation"])

    _render_greeks(result["greeks"], option_type)

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "Payoff",
        "Greeks vs Spot",
        "Price Heatmap",
    ])
    with chart_tab1:
        st.plotly_chart(
            _plot_payoff(
                spot,
                strike,
                selected_price,
                time_to_expiry,
                rate_pct / 100.0,
                sigma_pct / 100.0,
                dividend_pct / 100.0,
                option_type,
                currency,
            ),
            use_container_width=True,
        )
    with chart_tab2:
        st.plotly_chart(
            _plot_greek_sensitivity(
                spot,
                strike,
                time_to_expiry,
                rate_pct / 100.0,
                sigma_pct / 100.0,
                dividend_pct / 100.0,
                option_type,
            ),
            use_container_width=True,
        )
    with chart_tab3:
        st.plotly_chart(
            _plot_price_heatmap(
                strike,
                time_to_expiry,
                rate_pct / 100.0,
                option_type,
            ),
            use_container_width=True,
        )

    if result.get("implied_volatility") is not None:
        st.caption(
            f"Solved implied volatility from the observed market price: "
            f"{result['implied_volatility'] * 100.0:.2f}%."
        )
    else:
        confidence = result.get("confidence", confidence_score(spot, strike, time_to_expiry, sigma_pct / 100.0))
        st.caption(
            f"Confidence score without market-price comparison: {confidence.get('score', 0):.1f}/100 "
            f"({confidence.get('level', 'Low')})."
        )


def _render_live_market_analysis():
    st.markdown(
        '<div class="bs-note">Live or synthetic option-chain analysis using the shared engine. US chains come from yfinance; Indian stocks use a synthetic chain built from live spot plus historical volatility.</div>',
        unsafe_allow_html=True,
    )
    cfg1, cfg2, cfg3 = st.columns([2, 1, 1])
    with cfg1:
        symbol = st.text_input("Symbol", "AAPL", key="bs_live_symbol")
    with cfg2:
        expiry_days = st.number_input("Expiry (days for synthetic chains)", min_value=7, value=28, step=1, key="bs_live_expiry_days")
    with cfg3:
        threshold_pct = st.number_input("Signal Threshold (%)", min_value=1.0, value=5.0, step=0.5, key="bs_live_threshold")

    if not st.button("Fetch and Analyze", type="primary", key="bs_live_run"):
        return

    market_code = detect_market(symbol)
    currency = get_currency_symbol(market_code)
    rate = get_default_rate(market_code)

    with st.spinner(f"Loading option data for {symbol}..."):
        spot, hist_vol, dividend_yield = _fetch_spot_and_vol(symbol)
        if spot is None:
            st.error(f"Could not fetch price history for {symbol}.")
            return

        if market_code == "IN":
            calls_df, puts_df, expiry = _build_indian_option_chain(symbol, spot, hist_vol, rate, int(expiry_days))
            source_label = "Synthetic chain from live spot and historical volatility"
            time_to_expiry = int(expiry_days) / 365.0
        else:
            calls_df, puts_df, spot, expiry, hist_vol, dividend_yield = _fetch_us_options(symbol)
            if calls_df is None or puts_df is None:
                st.error("No live option chain was available for that symbol.")
                return
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
            time_to_expiry = max((expiry_dt - datetime.now()).days / 365.0, 1 / 365.0)
            source_label = "Live or delayed yfinance option chain"

    st.caption(
        f"Source: {source_label} | Spot: {currency}{spot:,.2f} | Historical Vol: {hist_vol * 100.0:.2f}% | Expiry: {expiry}"
    )

    tabs = st.tabs(["Calls", "Puts"])
    for tab, option_type, raw_df in zip(tabs, ["call", "put"], [calls_df, puts_df]):
        with tab:
            processed = _process_option_chain(
                raw_df,
                option_type,
                spot,
                time_to_expiry,
                rate,
                dividend_yield,
                threshold_pct,
                market_code,
            )
            if processed.empty:
                st.warning(f"No {option_type} contracts were suitable for analysis.")
                continue

            def _style_signal(val):
                return _signal_style(val)

            st.dataframe(
                processed.style.applymap(_style_signal, subset=["Signal"]),
                use_container_width=True,
                height=420,
            )
            st.download_button(
                label=f"Download {option_type.title()} analysis CSV",
                data=processed.to_csv(index=False),
                file_name=f"{symbol}_{option_type}_black_scholes.csv",
                mime="text/csv",
                key=f"bs_live_download_{option_type}",
            )


def _render_batch_analysis():
    st.markdown(
        '<div class="bs-note">Batch mode prices ATM options across multiple symbols using the same Black-Scholes-Merton engine and market-aware defaults.</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        symbols_raw = st.text_area(
            "Symbols (one per line)",
            "AAPL\nMSFT\nSPY\nRELIANCE.NS",
            height=160,
            key="bs_batch_symbols",
        )
    with col2:
        days_to_expiry = st.number_input("Days to Expiry", min_value=7, value=30, step=1, key="bs_batch_days")

    if not st.button("Run Batch Analysis", type="primary", key="bs_batch_run"):
        return

    rows = []
    symbols = [line.strip() for line in symbols_raw.splitlines() if line.strip()]
    if not symbols:
        st.warning("Enter at least one symbol.")
        return

    progress = st.progress(0, text="Starting batch analysis")
    for index, symbol in enumerate(symbols, start=1):
        progress.progress(index / len(symbols), text=f"Analyzing {symbol}")
        market_code = detect_market(symbol)
        currency = get_currency_symbol(market_code)
        rate = get_default_rate(market_code)
        spot, hist_vol, dividend_yield = _fetch_spot_and_vol(symbol)
        if spot is None:
            rows.append({"Symbol": symbol, "Status": "No history available"})
            continue

        time_to_expiry = days_to_expiry / 365.0
        result = analyze_option(
            S=spot,
            K=spot,
            T=time_to_expiry,
            r=rate,
            sigma=hist_vol,
            q=dividend_yield,
            option_type="call",
            market=market_code,
        )
        put_result = analyze_option(
            S=spot,
            K=spot,
            T=time_to_expiry,
            r=rate,
            sigma=hist_vol,
            q=dividend_yield,
            option_type="put",
            market=market_code,
        )
        call_greeks = result["greeks"]["call"]
        rows.append({
            "Symbol": symbol,
            "Market": market_code,
            "Spot": f"{currency}{spot:,.2f}",
            "Hist Vol %": round(hist_vol * 100.0, 2),
            "ATM Call": f"{currency}{result['theoretical_price']['call']:.2f}",
            "ATM Put": f"{currency}{put_result['theoretical_price']['put']:.2f}",
            "Delta": round(call_greeks["delta"], 4),
            "Gamma": round(call_greeks["gamma"], 4),
            "Theta": round(call_greeks["theta"], 4),
            "Vega": round(call_greeks["vega"], 4),
            "Rate %": round(rate * 100.0, 2),
        })
    progress.empty()

    batch_df = pd.DataFrame(rows)
    st.dataframe(batch_df, use_container_width=True, height=420)
    st.download_button(
        label="Download batch analysis CSV",
        data=batch_df.to_csv(index=False),
        file_name="black_scholes_batch_analysis.csv",
        mime="text/csv",
        key="bs_batch_download",
    )


def _render_csv_analysis():
    st.markdown(
        '<div class="bs-note">Upload a CSV with option scenarios to batch-price them through the shared engine.</div>',
        unsafe_allow_html=True,
    )
    sample = pd.DataFrame([
        {
            "symbol": "AAPL",
            "market": "US",
            "S": 185.0,
            "K": 190.0,
            "T_days": 30,
            "r": 0.053,
            "sigma": 0.28,
            "q": 0.006,
            "market_price": 6.5,
            "option_type": "call",
        },
        {
            "symbol": "RELIANCE.NS",
            "market": "IN",
            "S": 2950.0,
            "K": 3000.0,
            "T_days": 28,
            "r": 0.068,
            "sigma": 0.22,
            "q": 0.0,
            "market_price": 85.0,
            "option_type": "put",
        },
    ])
    st.download_button(
        label="Download sample CSV",
        data=sample.to_csv(index=False),
        file_name="black_scholes_template.csv",
        mime="text/csv",
        key="bs_csv_template",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="bs_csv_upload")
    if uploaded_file is None:
        return

    frame = pd.read_csv(uploaded_file)
    if "T_days" in frame.columns and "T" not in frame.columns:
        frame["T"] = frame["T_days"] / 365.0

    required = {"S", "K", "T", "r", "sigma"}
    missing = required - set(frame.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        return

    output_rows = []
    progress = st.progress(0, text="Running CSV analysis")
    for index, row in frame.iterrows():
        progress.progress((index + 1) / len(frame), text=f"Row {index + 1} of {len(frame)}")
        market_code = str(row.get("market", detect_market(str(row.get("symbol", "US"))))).upper()
        if market_code not in {"IN", "US"}:
            market_code = detect_market(str(row.get("symbol", "")))
        option_type = str(row.get("option_type", "call")).lower()
        market_price = float(row.get("market_price", 0.0) or 0.0)
        result = analyze_option(
            S=float(row["S"]),
            K=float(row["K"]),
            T=float(row["T"]),
            r=float(row["r"]),
            sigma=float(row["sigma"]),
            q=float(row.get("q", 0.0) or 0.0),
            market_price=market_price if market_price > 0 else None,
            option_type=option_type,
            market=market_code,
        )
        currency = get_currency_symbol(market_code)
        output_rows.append({
            "Symbol": row.get("symbol", f"row_{index + 1}"),
            "Market": market_code,
            "Option Type": option_type.upper(),
            "Call Price": f"{currency}{result['theoretical_price']['call']:.2f}",
            "Put Price": f"{currency}{result['theoretical_price']['put']:.2f}",
            "Delta": result["greeks"][option_type]["delta"],
            "Gamma": result["greeks"][option_type]["gamma"],
            "Theta": result["greeks"][option_type]["theta"],
            "Vega": result["greeks"][option_type]["vega"],
            "Rho": result["greeks"][option_type]["rho"],
            "Signal": result.get("valuation", {}).get("signal"),
            "Implied Vol %": (
                round(float(result["implied_volatility"]) * 100.0, 2)
                if result.get("implied_volatility") is not None
                else None
            ),
        })
    progress.empty()

    result_df = pd.DataFrame(output_rows)
    st.dataframe(result_df, use_container_width=True, height=420)
    st.download_button(
        label="Download CSV analysis results",
        data=result_df.to_csv(index=False),
        file_name="black_scholes_csv_results.csv",
        mime="text/csv",
        key="bs_csv_results_download",
    )


def render_black_scholes_analyzer():
    _inject_css()
    st.markdown("### Black-Scholes Analyzer")
    st.caption(
        "Uses the shared Black-Scholes-Merton engine from the backend, including dividend yield, parity checks, a Brent implied-vol solver, and market-aware defaults."
    )
    tabs = st.tabs([
        "Manual Analysis",
        "Live Market",
        "Batch Mode",
        "CSV Upload",
    ])
    with tabs[0]:
        _render_manual_analysis()
    with tabs[1]:
        _render_live_market_analysis()
    with tabs[2]:
        _render_batch_analysis()
    with tabs[3]:
        _render_csv_analysis()
