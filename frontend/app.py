import streamlit as st
import requests

st.title("Stock Predictor — Demo Frontend")

symbol = st.text_input("Stock symbol", "AAPL")

if st.button("Request prediction"):
    try:
        resp = requests.post("http://localhost:8000/enqueue/", json={"symbol": symbol})
        st.json(resp.json())
    except Exception as e:
        st.error(f"Request failed: {e}")
