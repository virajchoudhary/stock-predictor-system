import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Configuration
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Dashboard - QuantVision")

st.title("Market Trend Predictor")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, RELIANCE.NS)", "AAPL")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = {}

def analyze_stock():
    with st.spinner(f"Analyzing {symbol}..."):
        try:
            response = requests.get(f"{API_URL}/predict/{symbol}/")
            if response.status_code == 200:
                st.session_state.analysis_result = response.json()
            else:
                st.error("Failed to connect to backend.")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# Button triggers analysis, but UI renders from state
if st.button("Predict Trend"):
    analyze_stock()

# Render if data exists in state
if st.session_state.analysis_result:
    data = st.session_state.analysis_result
    if "error" in data:
        st.error(f"Error: {data['error']}")
    else:
        # --- Overview: always visible ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${data['current_price']:.2f}")
        with col2:
            pred_price = data['predicted_price']
            curr_price = data['current_price']
            pct_change = ((pred_price - curr_price) / curr_price) * 100
            st.metric("Predicted Price", f"${pred_price:.2f}", delta=f"{pct_change:.2f}%")
        with col3:
            if data['trend'] == "UP":
                st.success("BULLISH")
            elif data['trend'] == "DOWN":
                st.error("BEARISH")
            else:
                st.warning("NEUTRAL")
            if data.get('hyperparams_source') == 'evolved':
                st.info("Quantitative Model Active")
            else:
                st.warning("Default Model Active — optimizing overnight.")
        with col4:
            if data.get('hyperparams_source') == 'evolved':
                st.metric("Model Accuracy", f"{data.get('evolved_accuracy', 0):.1f}%")
            else:
                st.metric("Model Accuracy", "—")

        st.caption(f"Technical Signals: {', '.join(data.get('technicals', {}).get('signals', []))}")

        st.divider()
        st.subheader("AI Market Analysis")
        st.markdown(data.get('reasoning', 'No analysis available.'))

        # --- Technical Charts: collapsible ---
        with st.expander("Technical Charts", expanded=True):
            ohlc_json = data.get('ohlc_data', {})
            if ohlc_json:
                df_ohlc = pd.DataFrame(
                    ohlc_json['data'],
                    columns=ohlc_json['columns'],
                    index=pd.to_datetime(ohlc_json['index'])
                )
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_ohlc.index,
                    open=df_ohlc['Open'],
                    high=df_ohlc['High'],
                    low=df_ohlc['Low'],
                    close=df_ohlc['Close'],
                    name='Price'
                ))
                fig.update_layout(
                    xaxis_rangeslider_visible=True,
                    height=500,
                    title=f"{symbol} Price Action"
                )
                st.plotly_chart(fig, use_container_width=True)

                technicals = data.get('technicals', {})
                rsi_val = technicals.get('rsi', 50)
                st.metric("RSI (14)", f"{rsi_val:.2f}")
                st.progress(int(rsi_val))
                if rsi_val > 70:
                    st.warning("Overbought (>70)")
                elif rsi_val < 30:
                    st.success("Oversold (<30)")
                else:
                    st.info("Neutral")
            else:
                st.warning("No OHLC data for charts.")

            st.divider()
            st.subheader("Peer Comparison (% Return)")
            peer_history = data.get('peer_history', {})
            if peer_history:
                fig_comp = go.Figure()
                for s, prices in peer_history.items():
                    if prices:
                        start = prices[0]
                        pct = [((p - start) / start) * 100 for p in prices]
                        fig_comp.add_trace(go.Scatter(y=pct, mode='lines', name=s))
                fig_comp.update_layout(
                    yaxis_title="Return (%)",
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_comp, use_container_width=True)

        # --- Financials: collapsible ---
        with st.expander("Fundamentals", expanded=True):
            peer_financials = data.get('peer_financials', [])
            if peer_financials:
                df_fin = pd.DataFrame(peer_financials)
                cols = ['Symbol', 'Current Price', 'Market Cap', 'PE Ratio',
                        'EPS (Trailing)', 'Debt to Equity', 'Revenue Growth']
                final_cols = [c for c in cols if c in df_fin.columns]
                st.dataframe(df_fin[final_cols].style.highlight_max(axis=0),
                             use_container_width=True)
            else:
                st.info("No financial details.")

        # --- Chat: always at bottom ---
        st.divider()
        st.subheader("Ask about this Stock")
        
        if "dashboard_messages" not in st.session_state:
                st.session_state.dashboard_messages = []
        
        for msg in st.session_state.dashboard_messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if question := st.chat_input(f"Ask about {symbol}..."):
            # Context prep
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
            ] + [msg for msg in st.session_state.dashboard_messages if msg['role'] != 'system']

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_resp = requests.post(f"{API_URL}/chat/", json={"message": api_messages})
                        if chat_resp.status_code == 200:
                            ans = chat_resp.json().get("response", "Error")
                            st.markdown(ans)
                            st.session_state.dashboard_messages.append({"role": "assistant", "content": ans})
                        else:
                            st.error("Chat error")
                    except Exception as e:
                        st.error(f"Error: {e}")
