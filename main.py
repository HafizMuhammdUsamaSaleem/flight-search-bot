import json
import os
import logging
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv
import uuid
from typing import List
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not set in .env")
    raise ValueError("GOOGLE_API_KEY not set in .env")

MAX_FILE_SIZE = 10 * 1024 * 1024 
ALLOWED_EXTENSIONS = {".json", ".md"}

def validate_flight(flight: dict) -> bool:
    required_fields = {"airline", "alliance", "from", "to", "departure_date", "return_date", "layovers", "price_usd", "refundable"}
    return all(field in flight for field in required_fields)

def flight_to_doc(flight: dict) -> str:
    layovers_str = "no layovers" if not flight["layovers"] else f"layovers in {', '.join(flight['layovers'])}"
    refundable_str = "refundable" if flight["refundable"] else "non-refundable"
    return (
        f"Flight from {flight['from']} to {flight['to']} on {flight['airline']} ({flight['alliance']}). "
        f"Departure: {flight['departure_date']}, Return: {flight['return_date']}. "
        f"{layovers_str}. Price: ${flight['price_usd']} USD, {refundable_str}."
    )
@app.post("/create-embeddings")
async def create_embeddings(
    flights_file: UploadFile = File(...),
    visa_rules_file: UploadFile = File(...)
):
    start_time = time.time()
    logger.info("Starting create-embeddings endpoint")

    try:
        logger.debug(f"Received files: flights_file={flights_file.filename}, visa_rules_file={visa_rules_file.filename}")

        flights_ext = os.path.splitext(flights_file.filename)[1].lower()
        visa_ext = os.path.splitext(visa_rules_file.filename)[1].lower()
        logger.debug(f"Extracted extensions: flights_ext={flights_ext}, visa_ext={visa_ext}")
        if flights_ext not in ALLOWED_EXTENSIONS or visa_ext not in ALLOWED_EXTENSIONS:
            logger.error(f"Invalid file extensions: {flights_ext}, {visa_ext}")
            raise HTTPException(status_code=400, detail="Only .json and .md files are allowed")

        flights_content = await flights_file.read()
        visa_content = await visa_rules_file.read()
        if len(flights_content) > MAX_FILE_SIZE or len(visa_content) > MAX_FILE_SIZE:
            logger.error("File size exceeds 10MB limit")
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        if len(flights_content) == 0 or len(visa_content) == 0:
            logger.error("Empty file uploaded")
            raise HTTPException(status_code=400, detail="Uploaded files cannot be empty")
        try:
            flights = json.loads(flights_content.decode("utf-8"))
            if not isinstance(flights, list):
                logger.error("Flights file must contain a JSON array")
                raise HTTPException(status_code=400, detail="Flights file must contain a JSON array")
            for flight in flights:
                if not validate_flight(flight):
                    logger.error(f"Invalid flight data: {flight}")
                    raise HTTPException(status_code=400, detail=f"Invalid flight data: {flight}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in flights file: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in flights file: {e}")
        visa_rules = visa_content.decode("utf-8").split(". ")
        visa_rules = [rule.strip() + "." for rule in visa_rules if rule.strip()]
        docs = [flight_to_doc(f) for f in flights] + visa_rules
        logger.info(f"Processed {len(docs)} documents")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(docs, embeddings)
        vectorstore.save_local("faiss_index")
        with open("docs.txt", "w") as f:
            f.write("\n".join(docs))

        elapsed_time = time.time() - start_time
        logger.info(f"Vector store created in {elapsed_time:.2f} seconds")

        return {
            "status": "success",
            "message": "Vector store created and saved to faiss_index/",
            "document_count": len(docs),
            "processing_time_seconds": round(elapsed_time, 2)
        }
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")
    finally:
        await flights_file.close()
        await visa_rules_file.close()
def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not os.path.exists("faiss_index"):
            logger.error("FAISS index not found")
            raise FileNotFoundError("FAISS index not found. Run /create-embeddings first.")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store not found: {e}")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
prompt_template = """
You are a professional and courteous flight booking assistant designed to assist with travel-related inquiries. Your task is to provide accurate and relevant responses based on the retrieved context. Follow these instructions and guidelines strictly, adhering to the guardrails to ensure safe and appropriate interactions:

### Instructions:
- Use the provided context to answer questions accurately.
- Incorporate the chat history to maintain context and ensure coherent, context-aware responses across the conversation.
- Respond only when all required inputs (question, context, and chat history) are available; otherwise, politely decline with: "Insufficient data to respond. Please provide more details or check the context."

### Guidelines:
- If the question pertains to flights (e.g., routes, cheapest tickets, no layovers, specific dates, airlines), first check if the context provides matching details. If yes, deliver precise information.
- If the question is a flight or travel query but the context lacks specific matches (e.g., unavailable route like Dubai to KHI), respond transparently: "I don't have current details for flights from [origin] to [destination] in my data, which focuses on select international routes. For real-time options, check Emirates or PIA—direct flights often run $200-400 USD. How else can I assist with your trip?"
- If the question relates to travel policies or visas, provide relevant details from the context if available, or suggest: "Based on my data, [brief summary if possible]; for the latest, visit official embassy sites."
- If the question is truly off-topic or unrelated to travel (e.g., weather, recipes, personal advice), respond politely with: "I’m here to assist with flights and travel-related topics—please ask about tickets, visas, or other travel queries!"
- Keep responses concise, professional, and directly addressing the question, avoiding unnecessary elaboration, assumptions, or opinions. Always end with an open invitation like "What else can I help with?"
- If the context is insufficient for any reason, acknowledge it with: "I don’t have enough information to answer fully based on current data. Please rephrase or provide more details for better assistance."

### Guardrails:
- Do not generate content that is offensive, inappropriate, or outside the scope of travel assistance.
- Avoid speculative answers or fabricating details; if unsure, admit limitations and redirect to the user's travel focus.
- Do not store or reference personal user data beyond the current chat history.
- Terminate the conversation gracefully if "exit" is detected, responding with: "Session ended. Goodbye!" and ceasing further processing.

Chat history: {chat_history}
Question: {question}
Context: {context}

Provide your answer:
"""
prompt = PromptTemplate(input_variables=["chat_history", "question", "context"], template=prompt_template)

sessions = {}

def get_chain(memory):
    vectorstore = get_vectorstore()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

class Query(BaseModel):
    session_id: str | None = None
    question: str

@app.post("/query")
async def query(query: Query):
    try:
        if not query.session_id or query.session_id not in sessions:
            session_id = str(uuid.uuid4())
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            sessions[session_id] = memory
        else:
            session_id = query.session_id
            memory = sessions[session_id]

        chain = get_chain(memory)
        response = chain({"question": query.question})

        if "exit" in query.question.lower():
            del sessions[session_id]
            return {"session_id": session_id, "answer": "Session ended. Goodbye!", "history_cleared": True}

        return {"session_id": session_id, "answer": response["answer"]}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))