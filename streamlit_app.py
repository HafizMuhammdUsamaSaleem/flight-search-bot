import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import time
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not set in .env. Please add it to continue.")
    st.stop()
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="✈️ Flight RAG Assistant", page_icon="✈️", layout="wide")
st.markdown("""
<div style='text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px;'>
    <h1>✈️ Flight RAG Assistant</h1>
    <p style='color: #555;'>Your smart travel companion powered by Gemini, FAISS, and Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Session Control")
    if "session_id" in st.session_state:
        st.success(f"Active Session: `{st.session_state.session_id[:8]}...`")
        if st.button("End Session & Clear Chat", type="secondary"):
            st.session_state.clear()
            st.rerun()
    else:
        st.info("No active session. Start chatting to create one!")
    st.markdown("---")
    st.caption(f"Current time: November 05, 2025, 09:51 PM PKT")

def check_embeddings_exist():
    return os.path.exists("faiss_index")

embeddings_exist = check_embeddings_exist()

if not embeddings_exist:
    st.warning("📢 No knowledge base found! Upload your flight and visa files to start chatting.")
    st.markdown("### ✈️ Build Your Knowledge Base")
    st.info("Please upload `flights.json` (flight listings) and `visa_rules.md` (visa and policy rules) to create the knowledge base.")
    
    col1, col2 = st.columns(2)
    with col1:
        flights_file = st.file_uploader("**Flights Data (flights.json)**", type="json", help="JSON file with flight listings")
    with col2:
        visa_file = st.file_uploader("**Visa Rules (visa_rules.md)**", type="md", help="Markdown file with visa and ticket rules")

    if st.button("Build Knowledge Base 🚀", type="primary", use_container_width=True):
        if not flights_file or not visa_file:
            st.error("Please upload both `flights.json` and `visa_rules.md`!")
        else:
            with st.spinner("Processing..."):
                files = {
                    "flights_file": ("flights.json", flights_file.getvalue(), "application/json"),
                    "visa_rules_file": ("visa_rules.md", visa_file.getvalue(), "text/markdown")
                }
                try:
                    response = requests.post(f"{API_URL}/create-embeddings", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Knowledge base created! {result['message']}\nProcessed {result['document_count']} documents in {result['processing_time_seconds']} seconds.")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection failed: {e}\nEnsure `uvicorn main:app --reload` is running!")
else:
    st.success(" Knowledge base found! You can chat now or update it with new files.")
    
    with st.expander("Update Knowledge Base", expanded=False):
        st.markdown("### Upload New Files to Update Knowledge Base")
        st.info("Upload new `flights.json` and/or `visa_rules.md` to overwrite the existing knowledge base.")
        col1, col2 = st.columns(2)
        with col1:
            flights_file = st.file_uploader("**New Flights Data (flights.json)**", type="json", key="new_flights")
        with col2:
            visa_file = st.file_uploader("**New Visa Rules (visa_rules.md)**", type="md", key="new_visa")
        
        if st.button("Update Knowledge Base 🔄", type="primary", use_container_width=True):
            if not flights_file or not visa_file:
                st.error("Please upload both files to update!")
            else:
                with st.spinner("Processing..."): 
                    files = {
                        "flights_file": ("flights.json", flights_file.getvalue(), "application/json"),
                        "visa_rules_file": ("visa_rules.md", visa_file.getvalue(), "text/markdown")
                    }
                    try:
                        response = requests.post(f"{API_URL}/create-embeddings", files=files)
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Knowledge base updated! {result['message']}\nProcessed {result['document_count']} documents in {result['processing_time_seconds']} seconds.")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}\nEnsure `uvicorn main:app --reload` is running!")

    st.markdown("### 💬 Chat with Your Flight Assistant")
    st.info("Ask about flights (e.g., 'cheapest ticket from Dubai to Tokyo', 'no layovers', 'visa rules'). Type 'exit' to clear the session.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        try:
            response = requests.post(f"{API_URL}/query", json={"question": "hi"})
            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data["session_id"]
            else:
                st.error("Failed to start session. Is the API running?")
                st.stop()
        except Exception as e:
            st.error(f"Connection failed: {e}\nEnsure `uvicorn main:app --reload` is running!")
            st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about flights, visas, cheapest tickets, no layovers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "question": prompt
                    }
                    response = requests.post(f"{API_URL}/query", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        if "exit" in prompt.lower():
                            st.session_state.messages = []
                            st.session_state.pop("session_id", None)
                            st.success("Session cleared! Start a new chat or refresh.")
                            st.rerun()
                    else:
                        st.error(f"API Error: {response.json().get('detail', 'Unknown')}")
                except Exception as e:
                    st.error(f"Connection error: {e}\nEnsure `uvicorn main:app --reload` is running!")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #555;'>
    <p>Built with LangChain • Gemini • FAISS • Streamlit</p>
</div>
""", unsafe_allow_html=True)