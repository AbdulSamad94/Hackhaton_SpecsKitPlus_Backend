"""
RAG Helper Functions

Provides core functionality for the RAG pipeline:
- Text embedding generation using Google Gemini
- Vector similarity search in Qdrant
- RAG prompt building for context-aware responses
"""

import logging
from typing import List, Optional
from qdrant_client.http import models as rest
import google.generativeai as genai

logger = logging.getLogger(__name__)


def embed_text(text: str) -> List[float]:
    """Generate embedding for the given text using Google's embedding model."""
    result = genai.embed_content(
        model="models/text-embedding-004", content=text, task_type="retrieval_query"
    )
    embedding = result["embedding"]
    logger.info("Generated embedding with dimension: %d", len(embedding))
    return embedding


def build_selection_prompt(
    selected_text: str, user_question: str, contexts: List[dict] = None
) -> str:
    """
    Build a prompt for answering questions about user-selected text.

    Args:
        selected_text: The text the user selected from the textbook
        user_question: The user's question about the selected text
        contexts: Optional additional contexts from vector search

    Returns:
        Formatted prompt string for the AI agent
    """
    prompt_parts = [
        "You are helping a student understand a specific passage from the Physical AI & Humanoid Robotics textbook.",
        "",
        "SELECTED TEXT:",
        f'"""{selected_text.strip()}"""',
        "",
    ]

    # Optional: Include supplementary contexts if provided
    if contexts and len(contexts) > 0:
        prompt_parts.extend(["RELATED TEXTBOOK SECTIONS (for additional context):", ""])
        for i, ctx in enumerate(contexts, 1):
            prompt_parts.append(
                f"[Reference {i}: {ctx.get('title', '')} - {ctx.get('heading', '')}]\n{ctx.get('text', '')}\n"
            )
        prompt_parts.append("")

    # Add the question with clear framing
    prompt_parts.extend(
        [
            "STUDENT'S QUESTION:",
            user_question.strip(),
            "",
            "INSTRUCTIONS:",
            "- Answer the question by focusing on the SELECTED TEXT above",
            "- Provide clear, educational explanations suitable for a student",
            "- If the question cannot be fully answered from the selected text, acknowledge what information is missing",
            "- Keep your answer concise but complete",
        ]
    )

    return "\n".join(prompt_parts)
