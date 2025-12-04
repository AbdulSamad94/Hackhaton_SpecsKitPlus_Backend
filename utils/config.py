"""
Qdrant Vector Database Configuration

Initializes and configures connection to Qdrant Cloud for vector storage and retrieval.
"""

from qdrant_client import QdrantClient
import logging
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_qdrant() -> QdrantClient:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    logger.info("Connected to Qdrant at %s", QDRANT_URL)
    return client
