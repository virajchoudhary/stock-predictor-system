import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# Import quant_reporter
try:
    import quant_reporter as qr
    QUANT_REPORTER_AVAILABLE = True
except ImportError:
    QUANT_REPORTER_AVAILABLE = False

# Configuration
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Portfolio - QuantVision", page_icon="💼", layout="wide")

st.title("Portfolio Manager 💼")

# Tabs
tab_optimizer, tab_report = st.tabs(["🚀 Allocator (Optimizer)", "📄 Deep Report (Backtest)"])

# --- TAB 1: OPTIMIZER (Existing Logic) ---
with tab_optimizer:
    st.markdown("### Mean-Variance Optimization")
    st.write("Find the optimal asset allocation for a given risk tolerance.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_area("Enter Stock Tickers (comma separated)", "AAPL, GOOG, MSFT, TSLA", height=100)
    with col2:
        risk_tolerance = st.slider("Risk Tolerance (0=Safe, 1=Risky)", 0.0, 1.0, 0.5)

    if st.button("Optimize Portfolio", type="primary"):
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        
        if not tickers:
            st.warning("Please enter at least one ticker.")
        else:
            with st.spinner("Calculating optimal allocation..."):
                try:
                    response = requests.post(f"{API_URL}/optimize/", json={"tickers": tickers, "risk_tolerance": risk_tolerance})
                    if response.status_code == 200:
                        resp_json = response.json()
                        data = resp_json.get("allocation", {})
                        reasoning = resp_json.get("reasoning", "No analysis available.")
                        
                        st.success("Optimization Complete!")
                        st.subheader("Recommended Allocation")
                        
                        col_chart, col_table = st.columns(2)
                        
                        with col_chart:
                            # Create Pie Chart
                            labels = list(data.keys())
                            values = list(data.values())
                            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_table:
                            # Display Table
                            df = pd.DataFrame(list(data.items()), columns=["Ticker", "Weight"])
                            df["Weight"] = df["Weight"].apply(lambda x: f"{x:.1%}")
                            st.table(df)
                        
                        # AI Commentary
                        st.info(f"🤖 **AI Analysis:**\n\n{reasoning}")

                    else:
                        st.error("Optimization failed.")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# --- TAB 2: DEEP REPORT (Quant Reporter) ---
with tab_report:
    st.markdown("### Professional Quant Report")
    if not QUANT_REPORTER_AVAILABLE:
        st.error("`quant-reporter` library not installed. Please install it to use this feature.")
    else:
        st.write("Generate a comprehensive PDF-style HTML report with Backtesting, Efficient Frontier, and Monte Carlo simulations.")
        
        with st.expander("📝 Report Configuration", expanded=True):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                repo_tickers = st.text_input("Portfolio Tickers", "AAPL, MSFT, SPY, QQQ")
                benchmark = st.text_input("Benchmark Ticker", "SPY")
            with col_in2:
                # Dates
                default_start = datetime.now() - timedelta(days=365*2)
                start_date = st.date_input("Training Start Date", default_start)
                end_date = st.date_input("Training End Date", datetime.now() - timedelta(days=90)) # Leave 90 days for Test
                
        if st.button("Generate Deep Report 📊"):
            r_tickers = [t.strip() for t in repo_tickers.split(",") if t.strip()]
            
            if not r_tickers:
                 st.warning("Enter tickers.")
            else:
                # Create a simple equal-weight dict for the "User Portfolio" input
                # Real app would allow weight input, but equal weight is a good default for 'Analysis'
                weight = 1.0 / len(r_tickers)
                portfolio_dict = {t: weight for t in r_tickers}
                
                with st.spinner("Running Monte Carlo Simulations & Walk-Forward Analysis... (This may take 30s)"):
                    try:
                        import os
                        # Use absolute path to avoid cwd confusion
                        report_file = os.path.abspath("portfolio_report.html")
                        
                        # Call Quant Reporter
                        qr.create_combined_report(
                            portfolio_dict=portfolio_dict,
                            benchmark_ticker=benchmark,
                            train_start=start_date.strftime('%Y-%m-%d'),
                            train_end=end_date.strftime('%Y-%m-%d'),
                            filename=report_file
                        )
                        
                        if not os.path.exists(report_file):
                             raise FileNotFoundError(f"Report file was not created at {report_file}")

                        st.success(f"Report Generated Successfully at {report_file}!")
                        
                        # Read HTML content
                        with open(report_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # Display Download Button
                        st.download_button(
                            label="📥 Download HTML Report",
                            data=html_content,
                            file_name="My_Quant_Report.html",
                            mime="text/html"
                        )
                        
                        # Embed Report
                        st.divider()
                        st.subheader("Report Preview")
                        components.html(html_content, height=800, scrolling=True)
                        
                    except Exception as e:
                        st.error(f"Report Generation Failed: {e}")
