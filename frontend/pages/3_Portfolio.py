import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import threading
import os
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# Import quant_reporter
try:
    import quant_reporter as qr
    import yfinance as yf

    # --- MONKEY PATCH FOR quant_reporter TO HANDLE yfinance>=0.2.40 SERIES OBJECTS ---
    if hasattr(qr, 'opt_core'):
        def patched_get_risk_free_rate():
            try:
                tbill = yf.download("^IRX", period="5d")
                if tbill is None or tbill.empty:
                    raise Exception("^IRX download failed")
                latest_rate = float(np.ravel(tbill['Close'].iloc[-1])[0]) / 100
                if not 0 <= latest_rate <= 0.2:
                    raise Exception("Rate unrealistic.")
                return latest_rate
            except Exception:
                return 0.06
        qr.opt_core.get_risk_free_rate = patched_get_risk_free_rate

        def patched_calculate_rolling_returns(cumulative_df):
            periods = {'1-Year': 252, '3-Year': 252*3, '5-Year': 252*5}
            rolling_returns = {}
            for name, days in periods.items():
                if len(cumulative_df) > days:
                    years = days / 252
                    try:
                        end_val = float(np.ravel(cumulative_df.iloc[-1])[0])
                        start_val = float(np.ravel(cumulative_df.iloc[-days-1])[0])
                        val = (end_val / start_val) ** (1 / years) - 1
                        rolling_returns[name] = {0: f"{val:.2%}"}
                    except Exception:
                        rolling_returns[name] = {0: "N/A"}
                else:
                    rolling_returns[name] = {0: "N/A"}
            return pd.DataFrame.from_dict(rolling_returns, orient='index')

        qr.opt_core.calculate_rolling_returns = patched_calculate_rolling_returns

        # CRITICAL: also patch the reference inside combined_report module
        # because it imports the function directly into its own namespace
        try:
            import quant_reporter.combined_report as _qr_combined
            _qr_combined.calculate_rolling_returns = patched_calculate_rolling_returns
        except Exception:
            pass
    # ---------------------------------------------------------------------------------

    QUANT_REPORTER_AVAILABLE = True
except ImportError:
    QUANT_REPORTER_AVAILABLE = False

# Configuration
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Portfolio - QuantVision", layout="wide")

st.title("Portfolio Manager")

def _render_allocation_charts(allocation):
    if not allocation:
        st.warning("No allocation data.")
        return
    col_chart, col_table = st.columns(2)
    with col_chart:
        fig = go.Figure(data=[go.Pie(
            labels=list(allocation.keys()),
            values=list(allocation.values()),
            hole=0.4
        )])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, width='stretch')
    with col_table:
        df = pd.DataFrame(list(allocation.items()), columns=["Ticker", "Weight"])
        df["Weight"] = df["Weight"].apply(lambda x: f"{x:.1%}")
        st.table(df.set_index("Ticker"))

def _render_allocation(result, tickers):
    method = result.get("method", "standard")

    if method == "bl":
        # --- LSTM Views Table with animated spinners ---
        st.subheader("LSTM Model Views")
        st.caption(
            "Confidence is derived from each model's directional accuracy. "
            "Evolved = GA-optimized. Optimizing = GA running in background."
        )

        view_details = result.get("view_details", [])

        # Check for any queued/optimizing tickers
        has_optimizing = any(
            v.get("model_source") in ("queued", "default")
            for v in view_details
        )

        # Build custom HTML table with animated spinner for optimizing rows
        table_html = """
        <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
            margin: 0;
            padding: 8px;
            font-family: sans-serif;
        }
        .views-table {
            width: 100%;
            min-width: 1000px;
            border-collapse: collapse;
            font-size: 14px;
            margin-bottom: 16px;
        }
        .views-table th {
            text-align: left;
            padding: 8px 12px;
            border-bottom: 2px solid #333;
            color: #aaa;
            font-weight: 500;
        }
        .views-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #222;
            color: #fafafa;
        }
        .evolved-badge {
            color: #4CAF50;
            font-weight: 600;
        }
        .optimizing-badge {
            color: #FFA500;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .spinner {
            width: 12px;
            height: 12px;
            border: 2px solid #FFA500;
            border-top-color: transparent;
            border-radius: 50%;
            display: inline-block;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        </style>
        <table class="views-table">
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Current Price</th>
                <th>Predicted Price</th>
                <th>Expected Return</th>
                <th>Confidence</th>
                <th>Model</th>
                <th>Accuracy</th>
            </tr>
        </thead>
        <tbody>
        """

        for v in view_details:
            ret = v.get("expected_return", 0)
            ret_class = "positive" if ret >= 0 else "negative"
            ret_str = f"{ret:+.2f}%"

            source = v.get("model_source", "default")
            if source == "evolved":
                model_cell = f'<span class="evolved-badge">Evolved</span>'
            else:
                model_cell = (
                    '<span class="optimizing-badge">'
                    '<span class="spinner"></span>Optimizing...'
                    '</span>'
                )

            accuracy = v.get("accuracy")
            accuracy_str = f"{accuracy}%" if accuracy else "—"

            table_html += f"""
            <tr>
                <td><strong>{v['symbol']}</strong></td>
                <td>${v['current_price']}</td>
                <td>${v['predicted_price']}</td>
                <td class="{ret_class}">{ret_str}</td>
                <td>{v['confidence']}%</td>
                <td>{model_cell}</td>
                <td>{accuracy_str}</td>
            </tr>
            """

        table_html += "</tbody></table>"
        components.html(
            table_html,
            height=max(len(view_details) * 55 + 150, 300),
            scrolling=True
        )

        # Store optimizing status for auto-refresh
        st.session_state["has_optimizing"] = has_optimizing

        # Caption for queued tickers
        queued = [
            v["symbol"] for v in view_details
            if v.get("model_source") in ("queued", "default")
        ]
        if queued:
            st.caption(
                f"Optimizing in background: {', '.join(queued)}. "
                f"Page will refresh automatically when ready."
            )

        st.divider()
        st.subheader("Recommended Allocation")
        _render_allocation_charts(result.get("allocation", {}))
        st.caption("Method: AI-Driven Black-Litterman")

    else:
        # Standard fallback
        st.info("Using risk-based optimization (BL unavailable for these tickers).")
        st.subheader("Recommended Allocation")
        _render_allocation_charts(result.get("allocation", {}))
        st.caption("Method: Risk-Based (HRP/CVaR/Sharpe)")

    # AI Reasoning
    reasoning = result.get("reasoning", "")
    if reasoning:
        st.info(f"**AI Analysis:**\n\n{reasoning}")

def _run_allocation(tickers, risk_tolerance):
    """
    Tries BL first. Falls back to standard optimizer on failure.
    Stores result in session state.
    """
    with st.spinner("Running LSTM predictions and portfolio optimization..."):
        # Try Black-Litterman first
        try:
            bl_resp = requests.post(
                f"{API_URL}/bl-optimize/",
                json={"tickers": tickers},
                timeout=300
            )
            if bl_resp.status_code == 200:
                result = bl_resp.json()
                result["method"] = "bl"
                st.session_state["allocation_result"] = result
                _render_allocation(result, tickers)
                return
        except Exception:
            pass

        # Fallback to standard optimizer
        try:
            std_resp = requests.post(
                f"{API_URL}/optimize/",
                json={"tickers": tickers, "risk_tolerance": risk_tolerance}
            )
            if std_resp.status_code == 200:
                result = std_resp.json()
                result["method"] = "standard"
                st.session_state["allocation_result"] = result
                _render_allocation(result, tickers)
        except Exception as e:
            st.error(f"Optimization failed: {e}")

# Tabs
tab_allocator, tab_report = st.tabs([
    "Portfolio Allocator",
    "Deep Report (Backtest)"
])

# --- TAB 1: ALLOCATOR ---
with tab_allocator:
    # Silent auto-queue for default tickers on page load
    _default_tickers = ["AAPL", "TSLA", "SPY", "MSFT", "GOOG", "NVDA", "AMZN"]
    if "auto_queued" not in st.session_state:
        try:
            unoptimized = []
            for _t in _default_tickers:
                _r = requests.get(
                    f"{API_URL}/hpo-status/?symbol={_t}",
                    timeout=2
                )
                if _r.status_code == 200 and not _r.json().get("optimized"):
                    unoptimized.append(_t)
            if unoptimized:
                st.caption(
                    f"Models will be optimized automatically when you generate "
                    f"an allocation for: {', '.join(unoptimized)}"
                )
        except Exception:
            pass
        st.session_state["auto_queued"] = True

    st.markdown("### Portfolio Allocator")
    st.markdown(
        "Generates an optimal allocation using **AI-Driven Black-Litterman** "
        "optimization — combining LSTM price predictions with market equilibrium. "
        "Falls back to risk-based optimization if prediction data is unavailable."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_area(
            "Enter Stock Tickers (comma separated, min 2)",
            "AAPL, GOOG, MSFT, TSLA",
            height=80
        )
    with col2:
        risk_tolerance = st.slider("Risk Tolerance (0=Safe, 1=Risky)", 0.0, 1.0, 0.5)
        st.caption("Used as fallback if BL optimization fails.")

    if st.button("Generate Allocation", type="primary"):
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.warning("Enter at least 2 tickers.")
        else:
            st.session_state["last_tickers"] = tickers
            st.session_state["allocation_result"] = None
            st.session_state["refresh_count"] = 0
            st.session_state["_just_ran_allocation"] = True
            _run_allocation(tickers, risk_tolerance)

    # Re-render allocation result on tab switch (persists across reruns)
    if st.session_state.get("allocation_result") and \
       not st.session_state.get("_just_ran_allocation"):
        _render_allocation(
            st.session_state["allocation_result"],
            st.session_state.get("last_tickers", [])
        )

    # Reset flag after render
    st.session_state["_just_ran_allocation"] = False

    # Auto-refresh when tickers are still being optimized (non-blocking)
    if st.session_state.get("has_optimizing"):
        last_refresh = st.session_state.get("last_refresh_time", 0)
        current_time = time.time()
        if current_time - last_refresh >= 30:
            st.session_state["last_refresh_time"] = current_time
            last_tickers = st.session_state.get("last_tickers", [])
            still_optimizing = False
            for t in last_tickers:
                try:
                    r = requests.get(
                        f"{API_URL}/hpo-status/?symbol={t}",
                        timeout=3
                    )
                    if r.status_code == 200 and not r.json().get("optimized"):
                        still_optimizing = True
                        break
                except Exception:
                    pass
            if not still_optimizing:
                st.session_state["has_optimizing"] = False
            st.rerun()

# --- TAB 3: DEEP REPORT (Quant Reporter) ---
with tab_report:
    st.markdown("### Professional Quant Report")
    if not QUANT_REPORTER_AVAILABLE:
        st.error("`quant-reporter` library not installed. Please install it to use this feature.")
    else:
        st.write("Generate a comprehensive PDF-style HTML report with Backtesting, Efficient Frontier, and Monte Carlo simulations.")
        
        with st.expander("Report Configuration", expanded=True):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                repo_tickers = st.text_input("Portfolio Tickers", "AAPL, MSFT, SPY, QQQ")
                benchmark = st.text_input("Benchmark Ticker", "SPY")
            with col_in2:
                # Dates
                default_start = datetime.now() - timedelta(days=365*2)
                start_date = st.date_input("Training Start Date", default_start)
                end_date = st.date_input("Training End Date", datetime.now() - timedelta(days=90)) # Leave 90 days for Test
                
        if st.button("Generate Deep Report"):
            r_tickers = [t.strip() for t in repo_tickers.split(",") if t.strip()]
            if not r_tickers:
                st.warning("Enter tickers.")
            else:
                weight = 1.0 / len(r_tickers)
                portfolio_dict = {t: weight for t in r_tickers}
                report_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "portfolio_report.html"
                )
                report_file = os.path.normpath(report_file)
                with st.spinner("Generating report... (this may take 30-60 seconds)"):
                    try:
                        qr.create_combined_report(
                            portfolio_dict=portfolio_dict,
                            benchmark_ticker=benchmark,
                            train_start=start_date.strftime('%Y-%m-%d'),
                            train_end=end_date.strftime('%Y-%m-%d'),
                            filename=report_file
                        )
                        if os.path.exists(report_file):
                            with open(report_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.success("Report Generated Successfully!")
                            st.download_button(
                                label="Download HTML Report",
                                data=html_content,
                                file_name="My_Quant_Report.html",
                                mime="text/html"
                            )
                            st.divider()
                            st.subheader("Report Preview")
                            components.html(html_content, height=800, scrolling=True)
                        else:
                            st.error(f"Report file not created at {report_file}. Check Django terminal for errors.")
                    except Exception as e:
                        import traceback
                        st.error(f"Report Generation Failed: {e}")
                        st.code(traceback.format_exc())

