from typing import Optional
import logging
from dataclasses import dataclass
from agents import function_tool
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from utils.helpers import embed_text

logger = logging.getLogger(__name__)


@dataclass
class BookContext:
    """Represents a context chunk from the Physical AI textbook."""

    text: str
    title: str
    slug: str
    heading: str
    score: float


# Global references (initialized in main.py)
_qdrant_client: Optional[QdrantClient] = None


def set_qdrant_client(client: QdrantClient):
    """Set the global Qdrant client instance."""
    global _qdrant_client
    _qdrant_client = client


@function_tool
def search_book_content(query: str, top_k: int = 5, chapter_slug: Optional[str] = None):
    """
    Search the Physical AI textbook for relevant content based on a query.

    Args:
        query: The search query or question to find relevant content for
        top_k: Number of top results to return (default: 5)
        chapter_slug: Optional chapter slug to filter results to a specific chapter

    Returns:
        List of relevant text chunks with their metadata (title, heading, text, score)
    """
    if not _qdrant_client:
        raise RuntimeError("Qdrant client not initialized")

    logging.info(f"Searching book content for query: {query}")

    # Generate embedding for the query
    query_embedding = embed_text(query)

    # Build filter if chapter specified
    search_filter = None
    if chapter_slug:
        search_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="slug",
                    match=rest.MatchValue(value=chapter_slug),
                )
            ]
        )

    # Search Qdrant
    response = _qdrant_client.query_points(
        collection_name="physical_ai_book",
        query=query_embedding,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
        with_vectors=False,
    )

    # Format results
    contexts = []
    for point in response.points:
        payload = point.payload or {}
        contexts.append(
            BookContext(
                text=payload.get("text", ""),
                title=payload.get("title", ""),
                slug=payload.get("slug", ""),
                heading=payload.get("heading", ""),
                score=float(point.score) if point.score else 0.0,
            )
        )

    logging.info(f"Found {len(contexts)} relevant chunks")
    return contexts
