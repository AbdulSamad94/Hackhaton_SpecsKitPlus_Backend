# Backend API - RAG Chatbot

This is the FastAPI backend for the Physical AI & Humanoid Robotics textbook RAG chatbot.

## Architecture

- **Framework**: FastAPI
- **LLM**: Google Gemini (via OpenAI Agents SDK)
- **Vector Database**: Qdrant Cloud
- **Embeddings**: Google text-embedding-004

## Project Structure

```
backend/
├── main.py                 # FastAPI app with chat endpoints
├── models.py              # Agents SDK + Gemini configuration
├── ingest.py              # Script to index MDX files into Qdrant
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
└── utils/
    ├── config.py          # Qdrant client initialization
    ├── helpers.py         # Embedding, search, and prompt functions
    └── models.py          # Pydantic request/response models
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_key
   ```

3. Index documentation:
   ```bash
   python ingest.py
   ```

4. Run server:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### `POST /api/chat`
Chat with the RAG assistant.

**Request:**
```json
{
  "query": "What is ROS2?",
  "history": [],
  "top_k": 5,
  "chapter_slug": null
}
```

**Response:**
```json
{
  "answer": "ROS2 is...",
  "contexts": [...]
}
```

### `POST /api/ask-selection`
Ask questions about selected text.

**Request:**
```json
{
  "selected_text": "Text from the book...",
  "question": "What does this mean?"
}
```

**Response:**
```json
{
  "answer": "This refers to...",
  "selected_text": "...",
  "contexts": []
}
```

## Files

- **main.py**: FastAPI application with RAG logic
- **models.py**: Gemini + Agents SDK setup
- **ingest.py**: MDX document ingestion into Qdrant
- **utils/config.py**: Qdrant configuration
- **utils/helpers.py**: RAG helper functions
- **utils/models.py**: Request/response schemas
