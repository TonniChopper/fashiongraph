"""Reset the ChromaDB 'fashion_knowledge' collection.

Deletes the existing collection (if present) and creates a fresh,
empty one using a persistent client.

Usage::

    python -m src.scripts.reset_chroma
"""

import logging

import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

CHROMA_PATH: str = "./chroma_db"
COLLECTION_NAME: str = "fashion_knowledge"


def main() -> None:
    """Deletes and recreates the fashion_knowledge collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Check if collection exists and delete it
    existing: list[str] = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'.", COLLECTION_NAME)
    else:
        logger.info("Collection '%s' does not exist — nothing to delete.", COLLECTION_NAME)

    # Create a fresh empty collection
    client.create_collection(name=COLLECTION_NAME)
    logger.info("Created new empty collection '%s'.", COLLECTION_NAME)

    print("ChromaDB reset successfully")


if __name__ == "__main__":
    main()

