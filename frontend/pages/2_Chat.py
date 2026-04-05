import streamlit as st
import requests
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.style import inject_css, page_header

API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Advisor — QuantVis", page_icon="▲", layout="wide")

inject_css()
page_header("AI Financial Research", "Investment Advisor", accent="#00FF94")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about investment strategy, market analysis, or portfolio construction..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/chat/", json={"message": prompt})
                if response.status_code == 200:
                    ai_response = response.json().get("response", "No response received.")
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                else:
                    st.error("Failed to reach the advisor.")
            except Exception as e:
                st.error(f"Connection error: {e}")
