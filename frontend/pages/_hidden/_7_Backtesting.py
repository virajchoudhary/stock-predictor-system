import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# API Base URL
API_URL = "http://127.0.0.1:8000/api"

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Point-in-Time Backtesting - Stock Price Predictor",
    layout="wide",
)

st.title("Point-in-Time Historical Backtesting")
st.markdown("Test the LSTM model's exact forward predictions from any past date, preventing data leakage. Compares predicted trajectory against what actually happened.")

# ---------------------------------------------------------------------------
# Layout Inputs
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    symbol_input = st.text_input("Ticker Symbol", value="AAPL")

with col2:
    target_date = st.date_input(
        "Target Date (Prediction Origin)",
        value=datetime(2024, 7, 1),
        min_value=datetime(2022, 1, 1),
        max_value=datetime.today() - timedelta(days=90),
        help="The model will ONLY be trained on data prior to this date."
    )

with col3:
    st.write("") # Spacing
    st.write("") # Spacing
    force_retrain = st.checkbox(
        "Force Retrain",
        value=False,
        help="Purge cached hyperparameters and retrain from scratch."
    )

with col4:
    st.write("") # Spacing
    st.write("") # Spacing
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if run_btn:
    symbol = symbol_input.strip().upper()
    currency_symbol = "₹" if symbol.endswith((".NS", ".BO")) else "$"
    past_date_str = target_date.strftime("%Y-%m-%d")

    if not symbol:
        st.warning("Please enter a valid ticker symbol.")
    else:
        # --------------------------------------------------------------
        # Call both backtest endpoints
        # --------------------------------------------------------------
        lstm_data = None
        classical_data = None
        classical_error = None

        with st.spinner("Running LSTM + ARIMA/GARCH backtests..."):
            # Request 1: LSTM backtest (required)
            try:
                lstm_resp = requests.get(
                    f"{API_URL}/backtest-predict/{symbol}/",
                    params={
                        "past_date": past_date_str,
                        "force_retrain": "true" if force_retrain else "false",
                    },
                    timeout=120,
                )
                if lstm_resp.status_code != 200:
                    st.error(f"LSTM backtest error: {lstm_resp.text}")
                else:
                    candidate = lstm_resp.json()
                    if isinstance(candidate, dict) and "error" in candidate:
                        st.error(f"LSTM Backtest Error: {candidate['error']}")
                    else:
                        lstm_data = candidate
            except Exception as e:
                st.error(f"Could not reach LSTM endpoint. Is the backend running? ({e})")

            # Request 2: Classical ARIMA/GARCH backtest (additive, non-blocking)
            if lstm_data is not None:
                try:
                    classical_resp = requests.get(
                        f"{API_URL}/backtest-classical/{symbol}/",
                        params={"past_date": past_date_str},
                        timeout=120,
                    )
                    if classical_resp.status_code != 200:
                        classical_error = classical_resp.text
                    else:
                        candidate = classical_resp.json()
                        if isinstance(candidate, dict) and "error" in candidate:
                            classical_error = candidate["error"]
                        else:
                            classical_data = candidate
                except Exception as e:
                    classical_error = str(e)

        if lstm_data is None:
            st.stop()

        if classical_error is not None:
            st.warning(f"Classical model unavailable: {classical_error}")

        # --------------------------------------------------------------
        # Extract LSTM series
        # --------------------------------------------------------------
        lstm_hist = lstm_data.get("historical", {})
        lstm_test = lstm_data.get("test", {})
        lstm_metrics = lstm_data.get("metrics", {})

        hist_dates = [datetime.strptime(d, "%Y-%m-%d") for d in lstm_hist.get("dates", [])]
        hist_vals  = list(lstm_hist.get("prices", []))

        test_dates = [datetime.strptime(d, "%Y-%m-%d") for d in lstm_test.get("dates", [])]
        actual_vals = list(lstm_test.get("actual", []))
        lstm_pred_vals = list(lstm_test.get("predicted", []))

        # --------------------------------------------------------------
        # Extract Classical series (if available)
        # --------------------------------------------------------------
        classical_test = {}
        classical_metrics = {}
        classical_dates = []
        arima_pred_vals = []
        arima_lower = []
        arima_upper = []

        if classical_data is not None:
            classical_test = classical_data.get("test", {})
            classical_metrics = classical_data.get("metrics", {})
            classical_dates = [datetime.strptime(d, "%Y-%m-%d") for d in classical_test.get("dates", [])]
            arima_pred_vals = list(classical_test.get("predicted", []))
            arima_lower = list(classical_test.get("lower", []))
            arima_upper = list(classical_test.get("upper", []))

        # --------------------------------------------------------------
        # Build unified chart
        # --------------------------------------------------------------
        fig = go.Figure()

        # 1. Historical prices
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_vals,
            mode="lines",
            name="Historical",
            line=dict(color="#7A9BB5", width=1.5),
        ))

        # 2. GARCH confidence band (only if classical data available)
        if classical_data is not None and arima_upper and arima_lower and classical_dates:
            fig.add_trace(go.Scatter(
                x=classical_dates, y=arima_upper,
                mode="lines",
                line=dict(width=0),
                name="95% Confidence (GARCH)",
                showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=classical_dates, y=arima_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(191,127,255,0.12)",
                name="95% Confidence (GARCH) lower",
                showlegend=False,
            ))

        # 3. Actual test prices
        fig.add_trace(go.Scatter(
            x=test_dates, y=actual_vals,
            mode="lines",
            name="Actual Price",
            line=dict(color="#26A69A", width=2),
        ))

        # 4. LSTM predicted prices
        fig.add_trace(go.Scatter(
            x=test_dates, y=lstm_pred_vals,
            mode="lines",
            name="LSTM Predicted",
            line=dict(color="#F7A600", width=1.5, dash="dash"),
        ))

        # 5. ARIMA predicted prices
        if classical_data is not None and arima_pred_vals and classical_dates:
            fig.add_trace(go.Scatter(
                x=classical_dates, y=arima_pred_vals,
                mode="lines",
                name="ARIMA Predicted",
                line=dict(color="#BF7FFF", width=1.5, dash="dash"),
            ))

        # 6. Vertical marker at past_date
        all_prices = list(hist_vals) + list(actual_vals) + list(lstm_pred_vals) + list(arima_pred_vals) + list(arima_lower) + list(arima_upper)
        all_prices = [p for p in all_prices if p is not None]
        if all_prices:
            y_min = min(all_prices)
            y_max = max(all_prices)
            fig.add_trace(go.Scatter(
                x=[target_date, target_date],
                y=[y_min, y_max],
                mode="lines",
                name="Backtest Start",
                line=dict(color="#FF4466", width=2, dash="dot"),
            ))

        last_actual = lstm_data["test"]["actual"][-1] if lstm_data["test"]["actual"] else None
        if last_actual is not None:
            fig.add_hline(
                y=last_actual,
                line=dict(color="#26A69A", width=1, dash="dot"),
                annotation_text=f"Last Actual: {currency_symbol}{last_actual:.2f}",
                annotation_position="top right",
                annotation_font_color="#26A69A",
            )

        fig.update_layout(
            title=f"{symbol} — LSTM vs ARIMA/GARCH Backtest from {past_date_str}",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency_symbol})",
            paper_bgcolor="#080C10",
            plot_bgcolor="#0A0E14",
            font=dict(color="#E8F4FD"),
            height=520,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.01,
            ),
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#0A0E14",
                    activecolor="#F7A600",
                    font=dict(color="#E8F4FD"),
                ),
                rangeslider=dict(visible=False),
                type="date",
            ),
        )

        all_prices = []
        all_prices += lstm_data["historical"]["prices"]
        all_prices += lstm_data["test"]["actual"]
        all_prices += lstm_data["test"]["predicted"]
        if classical_data:
            all_prices += classical_data["test"]["predicted"]

        price_min = min(all_prices) * 0.97
        price_max = max(all_prices) * 1.03

        fig.update_layout(yaxis=dict(range=[price_min, price_max]))

        st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------------
        # Extended metrics
        # --------------------------------------------------------------
        st.subheader("Performance Metrics")

        st.markdown("**LSTM Model**")
        lc = st.columns(6)
        lc[0].metric("RMSE",           f"{lstm_metrics.get('rmse', 0):.2f}")
        lc[1].metric("MAE",            f"{lstm_metrics.get('mae', 0):.2f}")
        lc[2].metric("Win Rate",       f"{lstm_metrics.get('win_rate', 0):.1f}%")
        lc[3].metric("Sharpe Ratio",   f"{lstm_metrics.get('sharpe_ratio', 0):.3f}")
        lc[4].metric("Max Drawdown",   f"{lstm_metrics.get('max_drawdown', 0):.2f}%")
        lc[5].metric("Dir. Accuracy",  f"{lstm_metrics.get('directional_accuracy', 0):.1f}%")

        st.divider()

        st.markdown("**ARIMA + GARCH**")
        if classical_data is None:
            st.warning("Classical model did not run.")
        else:
            rc = st.columns(6)
            rc[0].metric("RMSE",          f"{classical_metrics.get('rmse', 0):.2f}")
            rc[1].metric("MAE",           f"{classical_metrics.get('mae', 0):.2f}")
            rc[2].metric("Win Rate",      f"{classical_metrics.get('win_rate', 0):.1f}%")
            rc[3].metric("Sharpe Ratio",  f"{classical_metrics.get('sharpe_ratio', 0):.3f}")
            rc[4].metric("Max Drawdown",  f"{classical_metrics.get('max_drawdown', 0):.2f}%")
            rc[5].metric("Dir. Accuracy", f"{classical_metrics.get('directional_accuracy', 0):.1f}%")
            arima_order = classical_metrics.get("arima_order", "N/A")
            st.caption(f"Auto-selected order: {arima_order}")

        # --------------------------------------------------------------
        # Equity curve chart
        # --------------------------------------------------------------
        st.subheader("Cumulative Strategy Returns (Normalized to 100)")

        eq_fig = go.Figure()

        lstm_eq = lstm_data.get("equity_curve", {})
        lstm_eq_dates = [datetime.strptime(d, "%Y-%m-%d") for d in lstm_eq.get("dates", [])]
        lstm_strategy = lstm_eq.get("lstm_equity") or lstm_eq.get("strategy_equity") or []
        buyhold       = lstm_eq.get("buyhold_equity", [])

        if lstm_strategy and lstm_eq_dates:
            eq_fig.add_trace(go.Scatter(
                x=lstm_eq_dates, y=lstm_strategy,
                mode="lines", name="LSTM Strategy",
                line=dict(color="#F7A600", width=2),
            ))
        if buyhold and lstm_eq_dates:
            eq_fig.add_trace(go.Scatter(
                x=lstm_eq_dates, y=buyhold,
                mode="lines", name="Buy & Hold",
                line=dict(color="#00FF94", width=1.5, dash="dash"),
            ))

        if classical_data is not None:
            cls_eq = classical_data.get("equity_curve", {})
            cls_eq_dates = [datetime.strptime(d, "%Y-%m-%d") for d in cls_eq.get("dates", [])]
            arima_strategy = cls_eq.get("arima_equity", [])
            if arima_strategy and cls_eq_dates:
                eq_fig.add_trace(go.Scatter(
                    x=cls_eq_dates, y=arima_strategy,
                    mode="lines", name="ARIMA Strategy",
                    line=dict(color="#BF7FFF", width=2),
                ))

        eq_fig.add_hline(
            y=100,
            line=dict(color="#7A9BB5", width=1, dash="dot"),
            annotation_text="Start (100)",
            annotation_position="bottom right",
            annotation_font_color="#7A9BB5",
        )

        eq_fig.update_layout(
            title="Cumulative Strategy Returns (Normalized to 100)",
            xaxis_title="Date",
            yaxis_title="Equity (base=100)",
            paper_bgcolor="#080C10",
            plot_bgcolor="#0A0E14",
            font=dict(color="#E8F4FD"),
            height=420,
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#0A0E14",
                    activecolor="#F7A600",
                    font=dict(color="#E8F4FD"),
                ),
                rangeslider=dict(visible=False),
                type="date",
            ),
        )

        st.plotly_chart(eq_fig, use_container_width=True)

        # --------------------------------------------------------------
        # Prediction log tables
        # --------------------------------------------------------------
        st.subheader("Prediction Log")

        def highlight_direction(val):
            if val is True:
                return "background-color: rgba(38,166,154,0.25); color: #26A69A"
            if val is False:
                return "background-color: rgba(255,68,102,0.25); color: #FF4466"
            return ""

        tab_lstm, tab_arima = st.tabs(["LSTM Predictions", "ARIMA Predictions"])

        with tab_lstm:
            lstm_trades = lstm_data.get("trade_log", [])
            if lstm_trades:
                df_l = pd.DataFrame(lstm_trades)
                styled_l = (
                    df_l.style
                    .applymap(highlight_direction, subset=["direction_correct"])
                    .format({
                        "predicted_price": f"{currency_symbol}{{:.2f}}",
                        "actual_price":    f"{currency_symbol}{{:.2f}}",
                        "error_abs":       f"{currency_symbol}{{:.2f}}",
                        "error_pct":       "{:.2f}%",
                    })
                )
                st.dataframe(styled_l, use_container_width=True, height=400)
            else:
                st.info("No LSTM trade log returned.")

        with tab_arima:
            if classical_data is None:
                st.info("Classical model did not run.")
            else:
                arima_trades = classical_data.get("trade_log", [])
                if arima_trades:
                    df_a = pd.DataFrame(arima_trades)
                    styled_a = (
                        df_a.style
                        .applymap(highlight_direction, subset=["direction_correct"])
                        .format({
                            "predicted_price": f"{currency_symbol}{{:.2f}}",
                            "actual_price":    f"{currency_symbol}{{:.2f}}",
                            "error_abs":       f"{currency_symbol}{{:.2f}}",
                            "error_pct":       "{:.2f}%",
                        })
                    )
                    st.dataframe(styled_a, use_container_width=True, height=400)
                else:
                    st.info("No ARIMA trade log returned.")

        st.caption("Each row is independently verifiable: predicted vs actual close, absolute/percent error, and direction-correctness.")
