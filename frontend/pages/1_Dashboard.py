import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.style import inject_css, page_header, section_label

API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Dashboard - Stock Price Predictor", page_icon="▲", layout="wide")

inject_css()
target_date = None
page_header("Market Intelligence", "Dashboard")

symbol = st.text_input("Symbol", "AAPL", placeholder="e.g. AAPL, RELIANCE.NS")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = {}


def analyze_stock():
    with st.spinner(f"Fetching data for {symbol}..."):
        try:
            params = {"target_date": target_date} if target_date else {}
            response = requests.get(f"{API_URL}/predict/{symbol}/", params=params)
            if response.status_code == 200:
                st.session_state.analysis_result = response.json()
            else:
                st.error("Backend unavailable.")
        except Exception as e:
            st.error(f"Connection error: {e}")


if st.button("Run Analysis", type="primary"):
    analyze_stock()

if st.session_state.analysis_result:
    data = st.session_state.analysis_result
    if "error" in data:
        st.error(data["error"])
    else:
        tab_overview, tab_charts, tab_financials, tab_validation = st.tabs(["OVERVIEW", "CHARTS", "FUNDAMENTALS", "MODEL VALIDATION"])

        with tab_overview:
            col1, col2, col3 = st.columns(3)
            currency_sym = "₹" if symbol.upper().endswith(".NS") or symbol.upper().endswith(".BO") else "$"
            with col1:
                st.metric("Current Price", f"{currency_sym}{data['current_price']:.2f}")
            with col2:
                pred_price = data["predicted_price"]
                curr_price = data["current_price"]
                pct_change = ((pred_price - curr_price) / curr_price) * 100
                st.metric("30-Day Forecast", f"{currency_sym}{pred_price:.2f}", delta=f"{pct_change:.2f}%")
            with col3:
                trend = data["trend"]
                trend_color = "#00FF94" if trend == "UP" else "#FF4466"
                st.markdown(f"""
                <div style="background:#0A0E14;border:1px solid #1E2D3D;
                border-top:2px solid {trend_color};padding:1rem 1.25rem;border-radius:3px;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                    letter-spacing:0.18em;text-transform:uppercase;color:#7A9BB5;margin-bottom:0.3rem;">
                    Trend Signal</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.3rem;color:{trend_color};">
                    {"BULLISH" if trend == "UP" else "BEARISH"}</div>
                </div>
                """, unsafe_allow_html=True)

            signals = data.get("technicals", {}).get("signals", [])
            if signals:
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#7A9BB5;'
                    f'letter-spacing:0.05em;margin-top:1rem;">Signals &nbsp;·&nbsp; '
                    f'{" &nbsp;·&nbsp; ".join(signals)}</div>',
                    unsafe_allow_html=True,
                )

            # ---- Backtest Verification (New Feature) ----
            verification = data.get("backtest_verification")
            if verification:
                actual_ret = verification["actual_return_30d"] * 100
                pred_ret   = ((data["predicted_price"] - data["current_price"]) / data["current_price"]) * 100
                
                # Success Logic: Did we get the direction right?
                success = (actual_ret > 0 and pred_ret > 0) or (actual_ret < 0 and pred_ret < 0)
                success_color = "#00FF94" if success else "#FF4466"
                
                st.markdown(f"""
                <div style="background:rgba(0,255,148,0.05);border:1px solid {success_color}33;
                border-left:4px solid {success_color};padding:1.25rem;border-radius:4px;margin:1.5rem 0;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                    letter-spacing:0.2em;text-transform:uppercase;color:#7A9BB5;margin-bottom:0.8rem;">
                    🎯 Backtest Verification (30-Day Actual Outcome)</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;">
                        <div>
                            <div style="font-size:0.58rem;color:#5BC0EB;text-transform:uppercase;margin-bottom:0.2rem;">Actual Return</div>
                            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;color:#E8F4FD;">{actual_ret:+.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size:0.58rem;color:#5BC0EB;text-transform:uppercase;margin-bottom:0.2rem;">Model Predicted</div>
                            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;color:#E8F4FD;">{pred_ret:+.2f}%</div>
                        </div>
                    </div>
                    <div style="margin-top:1rem;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:{success_color};">
                        {"✓ SUCCESS: Model correctly identified directional trend." if success else "✗ VARIANCE: Actual market conditions diverged from model projection."}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            section_label("AI Market Analysis", "#00D4FF")
            st.markdown(f"""
            <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.15);
            border-left:3px solid #00D4FF;padding:1rem 1.25rem;
            font-family:'IBM Plex Sans',sans-serif;font-size:0.83rem;color:#7A9BB5;line-height:1.75;">
            {data.get("reasoning", "No analysis available.")}
            </div>
            """, unsafe_allow_html=True)

        with tab_charts:
            ohlc_json = data.get("ohlc_data", {})
            if ohlc_json:
                df_ohlc = pd.DataFrame(
                    ohlc_json["data"],
                    columns=ohlc_json["columns"],
                    index=pd.to_datetime(ohlc_json["index"]),
                )

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_ohlc.index,
                    open=df_ohlc["Open"], high=df_ohlc["High"],
                    low=df_ohlc["Low"],   close=df_ohlc["Close"],
                    name="Price",
                    increasing_line_color="#00FF94",
                    decreasing_line_color="#FF4466",
                ))
                fig.update_layout(
                    paper_bgcolor="#080C10", plot_bgcolor="#080C10",
                    font=dict(family="IBM Plex Mono", color="#7A9BB5", size=10),
                    xaxis=dict(gridcolor="#1E2D3D", rangeslider_visible=True),
                    yaxis=dict(gridcolor="#1E2D3D"),
                    height=460,
                    title=dict(
                        text=f"{symbol} — Price Action",
                        font=dict(family="IBM Plex Mono", color="#E8F4FD", size=12),
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, width='stretch')

                technicals = data.get("technicals", {})
                rsi_val = technicals.get("rsi", 50)

                col_rsi, col_status = st.columns([1, 2])
                with col_rsi:
                    st.metric("RSI (14)", f"{rsi_val:.2f}")
                with col_status:
                    if rsi_val > 70:
                        st.warning("Overbought — RSI above 70")
                    elif rsi_val < 30:
                        st.success("Oversold — RSI below 30")
                    else:
                        st.info("Neutral — RSI within range")
                st.progress(int(rsi_val))
            else:
                st.warning("No OHLC data available.")

            st.divider()
            section_label("Peer Comparison — % Return")
            peer_history = data.get("peer_history", {})
            if peer_history:
                palette = ["#00D4FF", "#00FF94", "#FFB800", "#BF7FFF", "#FF4466"]
                fig_comp = go.Figure()
                for i, (s, prices) in enumerate(peer_history.items()):
                    if prices:
                        start = prices[0]
                        pct = [((p - start) / start) * 100 for p in prices]
                        fig_comp.add_trace(go.Scatter(
                            y=pct, mode="lines", name=s,
                            line=dict(color=palette[i % len(palette)], width=1.5),
                        ))
                fig_comp.update_layout(
                    paper_bgcolor="#080C10", plot_bgcolor="#080C10",
                    font=dict(family="IBM Plex Mono", color="#7A9BB5", size=10),
                    xaxis=dict(gridcolor="#1E2D3D"),
                    yaxis=dict(gridcolor="#1E2D3D", title="Return (%)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=300,
                )
                st.plotly_chart(fig_comp, width='stretch')

        with tab_financials:
            section_label("Fundamental Comparison")
            peer_financials = data.get("peer_financials", [])
            if peer_financials:
                df_fin = pd.DataFrame(peer_financials)
                cols = [
                    "Symbol", "Current Price", "Market Cap", "PE Ratio",
                    "EPS (Trailing)", "Debt to Equity", "Revenue Growth",
                ]
                final_cols = [c for c in cols if c in df_fin.columns]
                st.dataframe(
                    df_fin[final_cols].style.highlight_max(axis=0),
                    width='stretch',
                )
            else:
                st.info("No financial data available.")

        with tab_validation:
            st.markdown("### Model Validation (Proof of Accuracy)")
            st.markdown("Test the accuracy of the model by hiding recent data and predicting paths up to today.")
            from datetime import timedelta, datetime
            
            # Form layout
            col_date, col_cache, col_btn = st.columns([2, 2, 1])
            with col_date:
                past_date = st.date_input("Select Past Date", value=datetime.now() - timedelta(days=30), max_value=datetime.now() - timedelta(days=7), key="val_past_date")
            with col_cache:
                st.write("")  # spacing
                force_retrain = st.checkbox(
                    "Force Retrain (Ignore Cache)",
                    value=False,
                    help="Deletes stored hyperparameters for this symbol and retrains from scratch using the current architecture defaults.",
                )
            with col_btn:
                st.write("") # spacing
                st.write("") # spacing
                run_val = st.button("Run Validation", use_container_width=True)

            if run_val:
                spinner_msg = "Purging cache and retraining from scratch..." if force_retrain else "Truncating history and running LSTM validation slice..."
                with st.spinner(spinner_msg):
                    try:
                        force_retrain_param = "true" if force_retrain else "false"
                        val_resp = requests.get(
                            f"{API_URL}/backtest-predict/{symbol}/"
                            f"?past_date={past_date.strftime('%Y-%m-%d')}"
                            f"&force_retrain={force_retrain_param}"
                        )
                        if val_resp.status_code == 200:
                            val_data = val_resp.json()
                            if "error" in val_data:
                                st.error(val_data["error"])
                            else:
                                hist = val_data["historical"]
                                test = val_data["test"]
                                metrics = val_data["metrics"]

                                # Stitch lines together to avoid gaps
                                hist_dates = pd.to_datetime(hist["dates"])
                                hist_prices = hist["prices"]
                                
                                test_dates = pd.to_datetime(test["dates"])
                                actual_prices = test["actual"]
                                pred_prices = test["predicted"]
                                
                                if len(hist_dates) > 0 and len(test_dates) > 0:
                                    last_hist_date = hist_dates[-1]
                                    last_hist_price = hist_prices[-1]
                                    
                                    act_plot_dates = [last_hist_date] + list(test_dates)
                                    act_plot_prices = [last_hist_price] + actual_prices
                                    
                                    pred_plot_dates = [last_hist_date] + list(test_dates)
                                    pred_plot_prices = [last_hist_price] + pred_prices
                                else:
                                    act_plot_dates = list(test_dates)
                                    act_plot_prices = actual_prices
                                    pred_plot_dates = list(test_dates)
                                    pred_plot_prices = pred_prices

                                fig_val = go.Figure()
                                # 1. Historical Context Trace
                                fig_val.add_trace(go.Scatter(
                                    x=hist_dates, y=hist_prices,
                                    mode="lines", name="Historical",
                                    line=dict(color="#7A9BB5", dash="solid", width=2)
                                ))
                                # 2. Actual Market Path
                                fig_val.add_trace(go.Scatter(
                                    x=act_plot_dates, y=act_plot_prices,
                                    mode="lines", name="Actual Market Path",
                                    line=dict(color="#00FF94", dash="solid", width=2)
                                ))
                                # 3. Model Predicted Path
                                fig_val.add_trace(go.Scatter(
                                    x=pred_plot_dates, y=pred_plot_prices,
                                    mode="lines", name="LSTM Prediction",
                                    line=dict(color="#FFB800", dash="dash", width=2)
                                ))

                                fig_val.update_layout(
                                    paper_bgcolor="#080C10", plot_bgcolor="#080C10",
                                    font=dict(family="IBM Plex Mono", color="#7A9BB5", size=10),
                                    xaxis=dict(gridcolor="#1E2D3D", rangeslider_visible=True),
                                    yaxis=dict(gridcolor="#1E2D3D", title="Price"),
                                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    height=450,
                                )
                                st.plotly_chart(fig_val, use_container_width=True)

                                # Error Metrics row
                                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                                m_col1.metric("RMSE (Root Mean Square Error)", f"{metrics['rmse']:.2f}")
                                m_col2.metric("MAE (Mean Absolute Error)", f"{metrics['mae']:.2f}")
                                dir_acc = metrics['directional_accuracy']
                                dir_color = "normal" if dir_acc >= 50 else "inverse"
                                m_col3.metric("Directional Accuracy", f"{dir_acc:.1f}%", delta="Profitable Edge" if dir_acc >= 50 else "Random Walk", delta_color=dir_color)

                                # Expected Realized Return (Net of Fees) — styled red/green
                                err = metrics.get("expected_realized_return", 0.0)
                                err_color = "#00FF94" if err >= 0 else "#FF4466"
                                err_sign  = "+" if err >= 0 else ""
                                m_col4.markdown(
                                    f"""<div style="background:#0A0E14;border:1px solid #1E2D3D;
                                    border-top:2px solid {err_color};padding:1rem 1.25rem;border-radius:3px;">
                                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                                        letter-spacing:0.12em;text-transform:uppercase;color:#7A9BB5;margin-bottom:0.3rem;">
                                        Expected Realized Return (Net)</div>
                                        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                                        color:{err_color};font-weight:600;">{err_sign}{err:.2f}%</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                                st.caption(
                                    "Metrics account for T+1 Open execution, 0.1% commissions, "
                                    "and 0.05% slippage for real-world accuracy."
                                )
                        else:
                            st.error(f"Validation failed (Status: {val_resp.status_code})")
                    except Exception as e:
                        st.error(f"API Error: {e}")

        st.divider()
        section_label("Contextual Q&A", "#00D4FF")

        if "dashboard_messages" not in st.session_state:
            st.session_state.dashboard_messages = []

        for msg in st.session_state.dashboard_messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if question := st.chat_input(f"Ask about {symbol}..."):
            context = f"""
            User is looking at dashboard for {symbol}.
            Technical Signals: {data.get('technicals', {}).get('signals')}
            Trend: {data.get('trend')}
            Reasoning: {data.get('reasoning')}
            Financials: {data.get('financials')}
            """
            st.session_state.dashboard_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            api_messages = [
                {"role": "system", "content": f"You are a helpful financial assistant. Context:\n{context}"}
            ] + [msg for msg in st.session_state.dashboard_messages if msg["role"] != "system"]

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_resp = requests.post(f"{API_URL}/chat/", json={"message": api_messages})
                        if chat_resp.status_code == 200:
                            ans = chat_resp.json().get("response", "Error")
                            st.markdown(ans)
                            st.session_state.dashboard_messages.append({"role": "assistant", "content": ans})
                        else:
                            st.error("Chat error.")
                    except Exception as e:
                        st.error(f"Error: {e}")
