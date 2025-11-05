# flight-search-bot
An AI-powered travel app using FastAPI, Streamlit, and LangChain to query flights and visas , with RAG and FAISS for semantic search. 

## Project Overview
This is a **full-stack RAG (Retrieval-Augmented Generation) application** built for querying flight and travel information. It combines:
- **FastAPI** backend for API endpoints (embeddings creation and queries).
- **Streamlit** frontend for an interactive chat UI.
- **LangChain** for the RAG pipeline (embeddings, vector store, LLM chaining).
- **Sentence Transformers** for open-source embeddings.
- **FAISS** as the local vector database.
- **Google Gemini** as the LLM for generating responses.

The app processes flight data (`flights.json`) and visa rules (`visa_rules.md`), embeds them into a vector store, and allows users to query via natural language (e.g., "cheapest flight from Dubai to Tokyo in December"). It maintains chat history per session and handles off-topic queries gracefully.

**Key Features**:
- Upload and build knowledge base (flights + visas) dynamically.
- Semantic search for flights (e.g., by route, date, price, layovers).
- Context-aware responses with history (e.g., follow-ups like "any direct ones?").
- Generic "Processing..." spinner for all actions.
- Production-ready: File validation, logging, session management.
- UAE nationals-focused visa info for 5 destinations (Japan, USA, UK, France, Australia).

**Tech Stack**:
- Backend: FastAPI, LangChain, Gemini LLM, FAISS.
- Frontend: Streamlit.
- Embeddings: HuggingFace (all-MiniLM-L6-v2).
- Environment: Conda (Python 3.10).

**Use Cases**:
- Travel planning: "All options from Dubai to Tokyo in December."
- Visa checks: "Visa for UAE to Japan?"
- Policy queries: "Refund policy for Emirates."

## Repository Structure
```
├── streamlit_app.py
│── data
│    └── flights.json
│    └── visa_rules.md
└── faiss_index  
├── main.py
└── requirememts.txt
├── README.md
├── .env_example
    
```
## Setup Instructions
### 1. Clone & Setup Environment
```
git clone https://github.com/HafizMuhammdUsamaSaleem/flight-search-bot.git
cd flight-search-bot
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
```
### 2. Configure Environment
Copy `.env.example` to `.env` and add your Google API key

`GOOGLE_API_KEY=your_gemini_api_key_here`

### 3. Run the App

- Start FastAPI Backend (Terminal 1)
`uvicorn main:app --reload`
-Access docs at http://127.0.0.1:8000/docs (Swagger UI for testing endpoints).

-Start Streamlit Frontend (Terminal 2):
`streamlit run streamlit_app.py`

- Opens in browser at http://localhost:8501.

### 4. Build Knowledge Base (First Time)

- In Streamlit: Upload `data/flights.json`and `data/visa_rules.md`.
- Click "Build Knowledge Base " → See "Processing..." spinner.
- Success: Creates `faiss_index/` and refreshes to chat mode.

### 5. Interact

Chat Mode: Ask "cheapest flight from Dubai to Tokyo in December" or "visa for UAE to Japan?"
Update Data: Use the "Update Knowledge Base" expander to re-upload files.
Exit Session: Type "exit" → Clears history; sidebar has "End Session" button.
Test via Postman (see Endpoints section).

## How It Works: Step-by-Step Flow
### Overall RAG Pipeline

1. Data Ingestion: User uploads `flights.json` (25 Dubai-origin flights to 5 cities, future dates) and `visa_rules.md` (UAE visa policies + airline refunds).
2. Embedding Creation: Files converted to text docs (e.g., "Flight from Dubai to Tokyo on Emirates... Price: $850 USD"). Embeddings generated via Sentence Transformers and stored in FAISS.
3. Query Processing: User asks a question → Query embedded → Semantic search in FAISS retrieves top-3 matching docs → Fed to Gemini LLM with chat history and custom prompt.
4. Response Generation: LLM uses context for accurate, concise answers (e.g., lists flights by price/date). History ensures follow-ups work (e.g., "direct ones?").
5. Session Management: In-memory dict in FastAPI (keyed by session ID); Streamlit uses `st.session_state` for UI persistence (clears on "exit" or refresh).

### Streamlit Scenarios

1. No Embeddings Found (First Run or Deleted `faiss_index/`):

- Warning: "No knowledge base found!"
- Shows uploaders for `flights.json` and `visa_rules.md`.
- "Build Knowledge Base" button → "Processing..." spinner → API call to `/create-embeddings` → Success balloons + rerun to chat mode.


2. Embeddings Found (`faiss_index/` Exists):

Success: "Knowledge base found!"
Defaults to chat interface.
Expander for "Update Knowledge Base" → Optional re-upload + "Update" button → "Processing..." → Overwrites vector store.


3. Chat Flow:

- Displays history in chat bubbles.
- Input: Type query → "Processing..." spinner → API call to `/query` → Response appears.
- "Exit": Spinner brief → Clears messages/session → "Session cleared!" message.
- Off-Topic: Graceful redirect via prompt (e.g., "I don’t have details for that route...").


4. Error Handling: Connection issues show "Ensure uvicorn main:app --reload is running!" with red error boxes.

### Endpoints Details
`Access via http://127.0.0.1:8000/docs (Swagger UI).`

1. POST /create-embeddings (Build Vector Store):

- Purpose: Uploads files, validates (extensions: .json/.md, size <10MB, JSON structure), processes to docs, creates embeddings, saves FAISS index.
- Body: Multipart/form-data.
    - `flights_file:` JSON array of flights (required).
    - `visa_rules_file:` Markdown text (required).


- Response (200): `{"status": "success", "message": "...", "document_count": 30, "processing_time_seconds": 5.2}`.
- Errors: 400 (invalid files), 500 (server issues).
- Logs: Console shows processing time, doc count.


2. POST /query (Chat Query):

- Purpose: Embeds query, retrieves top-3 docs from FAISS, generates response with Gemini + history.
- Body: JSON.

    - `question`: String (required, e.g., "cheapest from Dubai to Tokyo").
    - `session_id`: String (optional; generates new if missing).


- Response (200): `{"session_id": "uuid", "answer": "The cheapest is Emirates for $850..."}`.
- Errors: 500 (no vector store: run /create-embeddings first).
- History: In-memory dict; clears on "exit".



### Custom Prompt
The LLM uses a structured prompt for smooth responses (e.g., handles unmatched routes like "Dubai to KHI" with: "I don't have details for Dubai to KHI... Check Emirates for $200-400 USD.").