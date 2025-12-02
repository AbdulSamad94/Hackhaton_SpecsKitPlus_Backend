from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from agents import Agent, Runner, set_tracing_disabled
from agents_instructions import main_agent_instruction
from pydantic_models import ChatResponse, ChatRequest
from models import model
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

set_tracing_disabled(True)

agent = Agent(name="Assistant", instructions=main_agent_instruction, model=model)


@app.get("/")
async def root():
    return {"message": "Chatbot API is running"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print(
            f"I received your message: '{request.message}'. This is a response from the FastAPI backend! 🚀"
        )
        result = await Runner.run(agent, request.message)

        return ChatResponse(response=result.final_output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
