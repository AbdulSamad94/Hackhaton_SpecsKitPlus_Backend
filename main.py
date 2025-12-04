"""
Physical AI Textbook RAG API

FastAPI application providing RAG-based chatbot endpoints for the Physical AI & Humanoid Robotics textbook.
Uses OpenAI Agents SDK with Google Gemini and Qdrant vector database.

Endpoints:
    - POST /api/chat: Chat with RAG assistant
    - POST /api/ask-selection: Ask questions about selected text
    - GET /api/health: Health check
"""

from fastapi import FastAPI, HTTPException
from agents import Agent, Runner, RunConfig
import os
import sys
import logging
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add parent directory to Python path for utils imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import init_qdrant
from utils.models import (
    ChatRequest,
    ChatResponse,
    AskSelectionRequest,
    AskSelectionResponse,
)
from utils.helpers import (
    embed_text,
    search_qdrant,
    build_rag_prompt,
    build_selection_prompt,
)
from models import model, client

logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Physical AI Textbook RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create RunConfig
run_config = RunConfig(model=model, model_provider=client, tracing_disabled=True)

print("[OK] OpenAI connection and LLM model initialized successfully!")


# ---------------------------
# Startup Event & Qdrant Client
# ---------------------------

# Global qdrant client (initialized lazily for serverless compatibility)
qdrant = None


def get_qdrant():
    """Get or initialize Qdrant client (serverless-compatible)"""
    global qdrant
    if qdrant is None:
        qdrant = init_qdrant()
        try:
            collection_info = qdrant.get_collection("physical_ai_book")
            logger.info(
                "Collection '%s' has %d points",
                "physical_ai_book",
                collection_info.points_count,
            )
            logger.info("Vector size: %s", collection_info.config.params.vectors)
        except Exception as e:
            logger.error("Failed to get collection info: %s", e)
    return qdrant


@app.on_event("startup")
def startup_event():
    """Initialize on startup for local development"""
    get_qdrant()
    logger.info("Startup complete.")


# ---------------------------
# Routes
# ---------------------------


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        query_emb = embed_text(req.query)
    except Exception as e:
        logger.exception("Failed to embed query: %s", e)
        raise HTTPException(status_code=500, detail="Embedding failed")

    try:
        # Get qdrant client (initializes if needed for serverless)
        qdrant_client = get_qdrant()
        points = search_qdrant(
            qdrant_client,
            query_emb,
            top_k=req.top_k,
            chapter_slug=req.chapter_slug,
        )
    except Exception as e:
        logger.exception("Qdrant search failed: %s", e)
        raise HTTPException(status_code=500, detail="Vector search failed")

    contexts: List[dict] = []
    for p in points:
        payload = p.payload or {}
        contexts.append(
            {
                "text": payload.get("text", ""),
                "title": payload.get("filename", ""),
                "slug": payload.get("source", ""),
                "heading": "",
                "score": p.score,
            }
        )

    prompt = build_rag_prompt(req.query, contexts)
    
    # Append history to prompt if needed
    history_context = ""
    if req.history:
        history_context = "\nChat History:\n"
        for msg in req.history[-5:]: # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_context += f"{role}: {content}\n"
    
    full_prompt = f"{history_context}\n{prompt}"

    try:
        # Create an agent with RAG context
        agent = Agent(
            name="Humanoid Robotics Textbook Assistant",
            instructions="You are a helpful tutor for a textbook about Physical AI & Humanoid Robotics. Answer questions using the provided context from the textbook.",
            model=model,
        )

        # Use the agent to generate answer with the RAG prompt
        result = await Runner.run(agent, full_prompt, run_config=run_config)
        answer = result.final_output

    except Exception as e:
        logger.exception("Agent generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Generation failed")

    return ChatResponse(answer=answer, contexts=contexts)


##### Ask On Selection Endpoint #####


@app.post("/ask-selection", response_model=AskSelectionResponse)
@app.post("/api/ask-selection", response_model=AskSelectionResponse)
async def ask_selection(req: AskSelectionRequest):
    """
    Answer contextual questions about user-selected text.
    """

    # Input validation
    if len(req.selected_text.strip()) < 5:
        raise HTTPException(
            status_code=400, detail="Selected text too short (minimum 5 characters)"
        )
    if len(req.selected_text.strip()) > 5000:
        raise HTTPException(
            status_code=400, detail="Selected text too long (maximum 5000 characters)"
        )
    if len(req.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Build prompt
    prompt = build_selection_prompt(req.selected_text, req.question)

    try:
        # Create agent with selection-focused instructions
        agent = Agent(
            name="Humanoid Robotics Textbook Assistant",
            instructions=(
                "You are a helpful tutor for a textbook about Physical AI & Humanoid Robotics. "
                "Help students understand selected passages by answering their questions clearly, "
                "accurately, and in a way that builds on the provided text context."
            ),
            model=model,
        )

        # Generate answer
        result = await Runner.run(agent, prompt, run_config=run_config)
        answer = result.final_output

    except Exception as e:
        logger.exception("Agent generation failed for ask-selection: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to generate answer. Please try again."
        )

    return AskSelectionResponse(
        answer=answer,
        selected_text=req.selected_text,
        contexts=[],
        metadata={
            "endpoint": "ask-selection",
            "question_length": len(req.question),
            "selection_length": len(req.selected_text),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
