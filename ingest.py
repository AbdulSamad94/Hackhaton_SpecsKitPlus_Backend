"""
MDX Document Ingestion Script

Indexes MDX documentation files from the docs/ directory into Qdrant vector database.
Uses Google Gemini text-embedding-004 for embeddings.

Usage:
    python ingest.py

This will:
    1. Scan all MDX files in docs/
    2. Clean frontmatter and imports
    3. Chunk text into 1000-char segments with 100-char overlap
    4. Generate embeddings using Gemini
    5. Upload to Qdrant collection 'physical_ai_book'
"""

import os
import glob
import re
import logging
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from uuid import uuid4

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs"))
COLLECTION_NAME = "physical_ai_book"

# Initialize Clients
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def clean_mdx(content: str) -> str:
    """Remove MDX frontmatter and some syntax."""
    # Remove frontmatter
    content = re.sub(r'^---[\s\S]*?---\n', '', content)
    # Remove import statements
    content = re.sub(r'^import .*?$', '', content, flags=re.MULTILINE)
    return content.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Simple text chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str) -> List[float]:
    """Get embedding from Gemini."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        return []

def ingest_docs():
    """Main ingestion function."""
    logger.info(f"Scanning docs in: {DOCS_DIR}")
    
    # Recreate collection
    try:
        qdrant.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE),
    )
    logger.info(f"Created collection: {COLLECTION_NAME}")

    mdx_files = glob.glob(os.path.join(DOCS_DIR, "**/*.md*"), recursive=True)
    logger.info(f"Found {len(mdx_files)} files")

    points = []
    
    for file_path in mdx_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            cleaned_content = clean_mdx(content)
            chunks = chunk_text(cleaned_content)
            
            rel_path = os.path.relpath(file_path, DOCS_DIR)
            filename = os.path.basename(file_path)
            
            logger.info(f"Processing {rel_path} - {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                embedding = get_embedding(chunk)
                if not embedding:
                    continue

                point_id = str(uuid4())
                payload = {
                    "text": chunk,
                    "source": rel_path,
                    "filename": filename,
                    "chunk_index": i
                }
                
                points.append(
                    rest.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                )
                
                # Batch upload every 50 points
                if len(points) >= 50:
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                    points = []
                    
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Upload remaining points
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    logger.info("Ingestion complete!")

if __name__ == "__main__":
    ingest_docs()
