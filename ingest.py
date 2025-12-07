"""
MDX Document Ingestion Script

Indexes MDX documentation files from the textbook/docs/ directory into Qdrant vector database.
Uses Google Gemini text-embedding-004 for embeddings.

Usage:
    python ingest.py

This will:
    1. Scan all MDX files in textbook/docs/
    2. Check against detailed cache (hash) to see if file changed
    3. Clean frontmatter and imports
    4. Chunk text into 1000-char segments with 100-char overlap
    5. Generate embeddings using Gemini
    6. Upload to Qdrant collection 'physical_ai_book'
"""

import os
import glob
import re
import logging
import json
import hashlib
from typing import List, Dict, Set
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from uuid import uuid4

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Fix: Point to the correct docs directory in textbook/docs
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../textbook/docs"))
COLLECTION_NAME = "physical_ai_book"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "ingestion_cache.json")

# Initialize Clients
if not all((GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY)):
    raise ValueError(
        "Missing one or more required environment variables: QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY"
    )
genai.configure(api_key=GEMINI_API_KEY)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def clean_mdx(content: str) -> str:
    """Remove MDX frontmatter and some syntax."""
    # Remove frontmatter
    content = re.sub(r"^---[\s\S]*?---\n", "", content)
    # Remove import statements
    content = re.sub(r"^import .*?$", "", content, flags=re.MULTILINE)
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


def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_cache() -> Dict[str, str]:
    """Load ingestion cache."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load or parse cache file: {e}")
    return {}


def save_cache(cache: Dict[str, str]):
    """Save ingestion cache."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save cache file: {e}")


def ingest_docs():
    """Main ingestion function."""
    logger.info(f"Scanning docs in: {DOCS_DIR}")

    # Ensure collection exists
    try:
        # Check if collection exists first
        qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' exists.")
    except Exception:
        # Create if not exists
        logger.info(f"Collection '{COLLECTION_NAME}' not found. Creating...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE),
        )

    # Ensure payload index exists for 'source' (required for delete operations)
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created payload index for 'source'")
    except Exception as e:
        logger.warning(f"Note on creating index: {e}")

    mdx_files = glob.glob(os.path.join(DOCS_DIR, "**/*.md*"), recursive=True)
    logger.info(f"Found {len(mdx_files)} files")

    cache = load_cache()
    new_cache = cache.copy()
    files_processed = 0
    points_uploaded = 0

    for file_path in mdx_files:
        try:
            rel_path = os.path.relpath(file_path, DOCS_DIR).replace(os.sep, "/")

            # Check if file has changed
            current_hash = calculate_file_hash(file_path)
            if rel_path in cache and cache[rel_path] == current_hash:
                logger.debug(f"Skipping unmodified file: {rel_path}")
                continue

            logger.info(f"Processing changed file: {rel_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            cleaned_content = clean_mdx(content)
            chunks = chunk_text(cleaned_content)

            filename = os.path.basename(file_path)
            points = []

            # First, delete existing points for this file to avoid duplicates
            # (In a real production system we might use point IDs based on hash,
            # but deleting by payload filter is safer for updates)
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=rest.FilterSelector(
                    filter=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="source",
                                match=rest.MatchValue(value=rel_path),
                            )
                        ]
                    )
                ),
            )

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
                    "chunk_index": i,
                }

                points.append(
                    rest.PointStruct(id=point_id, vector=embedding, payload=payload)
                )

            # Upload points for this file
            if points:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                points_uploaded += len(points)

            # Update cache
            new_cache[rel_path] = current_hash
            files_processed += 1

            # Save cache periodically
            if files_processed % 5 == 0:
                save_cache(new_cache)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Final cache save
    save_cache(new_cache)

    logger.info(
        f"Ingestion complete! Processed {files_processed} files, uploaded {points_uploaded} chunks."
    )


if __name__ == "__main__":
    ingest_docs()
