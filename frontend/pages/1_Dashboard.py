import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Configuration
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Dashboard - QuantVision", page_icon="📊")

st.title("Market Trend Predictor 📊")

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
        # --- Tabs Layout ---
        tab_overview, tab_charts, tab_financials, tab_news = st.tabs(["Overview", "Technical Charts", "Financials", "News"])

        with tab_overview:
            # 1. Top Section: Price & Trend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${data['current_price']:.2f}")
            with col2:
                pred_price = data['predicted_price']
                curr_price = data['current_price']
                pct_change = ((pred_price - curr_price) / curr_price) * 100
                st.metric("Predicted Price (30d)", f"${pred_price:.2f}", delta=f"{pct_change:.2f}%")
            with col3:
                st.header(f"Trend: {data['trend']}")
                if data['trend'] == "UP":
                    st.success("BULLISH 🚀")
                else:
                    st.error("BEARISH 📉")
            
            st.caption(f"Technical Signals: {', '.join(data.get('technicals', {}).get('signals', []))}")
            
            # 2. AI Analysis
            st.divider()
            st.subheader("🤖 AI Market Analysis")
            st.markdown(data.get('reasoning', 'No analysis available.'))

        with tab_charts:
            st.subheader("Interactive Price Chart")
            ohlc_json = data.get('ohlc_data', {})
            if ohlc_json:
                 # Reconstruct DataFrame
                 df_ohlc = pd.DataFrame(ohlc_json['data'], columns=ohlc_json['columns'], index=pd.to_datetime(ohlc_json['index']))
                 
                 # Create Candlestick Chart with Volume
                 fig = go.Figure()
                 fig.add_trace(go.Candlestick(x=df_ohlc.index,
                                 open=df_ohlc['Open'],
                                 high=df_ohlc['High'],
                                 low=df_ohlc['Low'],
                                 close=df_ohlc['Close'],
                                 name='Price'))
                 
                 fig.update_layout(xaxis_rangeslider_visible=True, height=500, title=f"{symbol} Price Action")
                 st.plotly_chart(fig, use_container_width=True)
                 
                 # RSI Indicator
                 technicals = data.get('technicals', {})
                 rsi_val = technicals.get('rsi', 50)
                 
                 st.metric("RSI (14)", f"{rsi_val:.2f}")
                 st.progress(int(rsi_val))
                 if rsi_val > 70: st.warning("Overbought (>70)")
                 elif rsi_val < 30: st.success("Oversold (<30)")
                 else: st.info("Neutral")
                 
            else:
                st.warning("No OHLC data for charts.")

            st.divider()
            st.subheader("📈 Peer Comparison (% Return)")
            peer_history = data.get('peer_history', {})
            if peer_history:
                fig_comp = go.Figure()
                for s, prices in peer_history.items():
                    if prices:
                        start = prices[0]
                        pct = [((p-start)/start)*100 for p in prices]
                        fig_comp.add_trace(go.Scatter(y=pct, mode='lines', name=s))
                fig_comp.update_layout(yaxis_title="Return (%)", margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_comp, use_container_width=True)

        with tab_financials:
            st.subheader("📊 Fundamental Comparison")
            peer_financials = data.get('peer_financials', [])
            if peer_financials:
                df_fin = pd.DataFrame(peer_financials)
                cols = ['Symbol', 'Current Price', 'Market Cap', 'PE Ratio', 'EPS (Trailing)', 'Debt to Equity', 'Revenue Growth']
                final_cols = [c for c in cols if c in df_fin.columns]
                st.dataframe(df_fin[final_cols].style.highlight_max(axis=0))
            else:
                st.info("No financial details.")

        with tab_news:
             st.subheader("📰 Recent News")
             news = data.get('news', [])
             for n in news:
                 with st.expander(f"{n.get('title')} - {n.get('publisher')}"):
                     st.markdown(f"[Read Article]({n.get('link')})")

        # 6. Contextual Chat (Always visible at bottom)
        st.divider()
        st.subheader("💬 Ask about this Stock")
        
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
