import streamlit as st
import requests
import os

# Configuration
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="QuantVision",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("QuantVision")
st.subheader("AI-Driven Intraday Market Trend Predictor & Investment Advisor")

st.markdown("""
Welcome to **QuantVision**. This app leverages advanced AI to provide:
*   **Intraday Trend Predictions** for Indian Equities.
*   **Personalized Portfolio Optimization**.
*   **AI Financial Advisor Chatbot**.
*   **Black–Scholes Options Analyzer** — theoretical pricing, Greeks, implied volatility, and trading signals.

Navigate using the sidebar to access different features.
""")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info("Select a module from the pages above.")
