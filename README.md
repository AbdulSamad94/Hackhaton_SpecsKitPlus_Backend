# Backend API - Agentic RAG Chatbot

FastAPI backend for the Physical AI & Humanoid Robotics textbook using Agentic RAG with the OpenAI Agents SDK.

## Architecture

- **Framework**: FastAPI
- **LLM**: Google Gemini (via OpenAI Agents SDK)
- **Vector Database**: Qdrant Cloud
- **Embeddings**: Google text-embedding-004
- **RAG Approach**: **Agentic RAG** - The AI agent autonomously decides when to search the textbook using function tools

## Key Features

### 🤖 Agentic RAG with Function Tools

Instead of manually injecting context, the agent uses the `search_book_content` tool to autonomously search the textbook when needed.

### 📚 Incremental Ingestion

Smart caching system that only re-indexes modified files, saving API costs:

- **First run**: Indexes all documents
- **Subsequent runs**: Skips unmodified files (0 API calls)
- **After edits**: Only re-indexes changed files

### 🎯 User Personalization

Customizes responses based on user's software/hardware background from their profile.

### 💬 Markdown Formatting

AI generates responses in clean markdown format for better readability.

## Project Structure

```
backend/
├── main.py                 # FastAPI app with Agentic RAG endpoints
├── models.py              # Agents SDK + Gemini configuration
├── personalization.py     # User context-aware system prompts
├── ingest.py              # Incremental MDX document ingestion
├── ingestion_cache.json   # File hashes for incremental updates
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
└── utils/
    ├── config.py          # Qdrant client initialization
    ├── helpers.py         # Embedding & prompt functions
    ├── models.py          # Pydantic request/response schemas
    └── tools.py           # Function tools for agent (search_book_content)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Index Documentation

Run the ingestion script to index the textbook:

```bash
python ingest.py
```

This will:

- Scan all `.md` and `.mdx` files in `../textbook/docs`
- Chunk text into 1000-character segments
- Generate embeddings using Gemini
- Upload to Qdrant collection `physical_ai_book`
- Create `ingestion_cache.json` for incremental updates

**Note**: Running it again will skip unmodified files (0 API usage).

### 4. Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

## API Endpoints

### `POST /api/chat`

Chat with the Agentic RAG assistant.

**Request:**

```json
{
  "query": "What is ROS2?",
  "history": [
    { "role": "user", "content": "Previous message" },
    { "role": "bot", "content": "Previous response" }
  ],
  "user_context": {
    "software_background": "Python, C++",
    "hardware_background": "Arduino, Raspberry Pi"
  }
}
```

**Response:**

```json
{
  "answer": "## ROS 2\n\nROS 2 is the next generation...",
  "contexts": []
}
```

**Note**: `contexts` is empty in Agentic RAG because the tool handles retrieval internally.

### `POST /api/ask-selection`

Ask questions about user-selected text from the textbook.

**Request:**

```json
{
  "selected_text": "Text highlighted by user...",
  "question": "What does this mean?",
  "user_context": {
    "software_background": "JavaScript",
    "hardware_background": null
  }
}
```

**Response:**

```json
{
  "answer": "This passage explains...",
  "selected_text": "...",
  "contexts": []
}
```

### `GET /api/health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

## How Agentic RAG Works

1. **User Query** → Sent to `/api/chat`
2. **Agent Creation** → Agent is initialized with:
   - Personalized system instructions
   - `search_book_content` tool
3. **Autonomous Decision** → Agent decides whether to search the textbook
4. **Tool Execution** → If needed, agent calls `search_book_content(query="...")`
5. **Response Generation** → Agent synthesizes answer from tool results
6. **Markdown Formatting** → Response formatted with headings, lists, bold text

## Development

### Running Tests

```bash
# Add test commands when available
```

### Deployment

Deploy to Vercel, Railway, or any platform supporting FastAPI.

**Note**: Set environment variables in your deployment platform.

## Troubleshooting

### Ingestion Errors

- **"Index required but not found"**: The script automatically creates payload indexes. If you see this, delete the collection manually and re-run `ingest.py`.
- **"Collection not found"**: Run `python ingest.py` to create and populate the collection.

### API Errors

- **500 errors**: Check your `.env` file has valid API keys.
- **Embedding failed**: Verify `GEMINI_API_KEY` is correct.
- **Qdrant search failed**: Verify `QDRANT_URL` and `QDRANT_API_KEY`.

## File Reference

- **main.py**: FastAPI routes, agent initialization, tool registration
- **personalization.py**: Generates personalized system prompts with formatting guidelines
- **models.py**: Gemini + OpenAI Agents SDK configuration
- **ingest.py**: Document chunking, embedding, and incremental indexing
- **utils/config.py**: Qdrant client setup
- **utils/helpers.py**: Embedding generation, Qdrant search, prompt builders
- **utils/models.py**: Pydantic schemas for API requests/responses
- **utils/tools.py**: `search_book_content` function tool for agent
