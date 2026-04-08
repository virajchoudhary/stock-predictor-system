import streamlit as st
import requests

# Configuration
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="AI Chat - QuantVision")

st.title("AI Investment Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask for investment advice..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Backend call
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/chat/", json={"message": prompt})
                if response.status_code == 200:
                    ai_response = response.json().get("response", "No response received.")
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                else:
                    st.error("Failed to get response from AI.")
            except Exception as e:
                st.error(f"Connection Error: {e}")
