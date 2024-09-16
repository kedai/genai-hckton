# app.py
import streamlit as st
from langchain_setup import run_agent

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.set_page_config(page_title="GenAI Exchange Rate Analyzer", layout="wide")

st.title("GenAI Exchange Rate Analyzer")

menu = ["Home", "Rates by Date", "Plot Trends", "Ask AI"]
choice = st.sidebar.selectbox("Menu", menu)

st.subheader("Ask the AI Assistant")

# Display chat history
for message in st.session_state['messages']:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            if "data:image/png;base64" in message["content"]:
                # Display the image
                st.image(message["content"], use_column_width=True)
            else:
                st.markdown(message["content"])

# Input for user to type their question
if prompt := st.chat_input("Enter your question here"):
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Processing..."):
        try:
            # Get response from the agent
            response = run_agent(prompt)
            # Check if response contains a base64 image
            if response.startswith("data:image/png;base64,"):
                st.session_state['messages'].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.image(response, use_column_width=True)
            else:
                st.session_state['messages'].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.session_state['messages'].append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)
