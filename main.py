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
from utils.helpers import build_selection_prompt
from utils.tools import search_book_content, set_qdrant_client
from models import model, client
from personalization import create_personalized_prompt

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
    # Ensure Qdrant client is available for tools
    # This must be called before the tool is invoked
    set_qdrant_client(get_qdrant())

    # Example of how we might pass history if the Agent supports it in the string prompt:
    history_context = ""
    if req.history:
        history_context = "\nChat History:\n"
        for msg in req.history[-5:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_context += f"{role}: {content}\n"

    try:
        # Generate personalized system instruction
        system_instruction = create_personalized_prompt(req.user_context)

        # Create an agent with the 'search_book_content' tool
        agent = Agent(
            name="Humanoid Robotics Textbook Assistant",
            instructions=system_instruction,
            model=model,
            tools=[search_book_content],
        )

        # Run the agent
        # It will autonomously decide to call 'search_book_content' if needed
        result = await Runner.run(
            agent, f"{history_context}\nUser Query: {req.query}", run_config=run_config
        )
        answer = result.final_output

    except Exception as e:
        logger.exception("Agent generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Generation failed")

    # In Agentic RAG, we don't manually return 'contexts' list because the tool
    # handles retrieval internally. The UI might need adjustment or we accept empty contexts.
    return ChatResponse(answer=answer, contexts=[])


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
        # Generate personalized system instruction
        base_instruction = (
            "You are a helpful tutor for a textbook about Physical AI & Humanoid Robotics. "
            "Help students understand selected passages by answering their questions clearly, "
            "accurately, and in a way that builds on the provided text context."
        )

        # Use a personalized prompt if background information is available.
        if req.user_context and (
            req.user_context.software_background or req.user_context.hardware_background
        ):
            system_instruction = create_personalized_prompt(req.user_context)
        else:
            system_instruction = base_instruction

        # Create agent with selection-focused instructions
        agent = Agent(
            name="Humanoid Robotics Textbook Assistant",
            instructions=system_instruction,
            model=model,
            tools=[search_book_content],
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
