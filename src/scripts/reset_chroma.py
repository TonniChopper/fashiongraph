"""Reset the ChromaDB 'fashion_knowledge' collection.

Completely wipes the ChromaDB directory on disk and recreates a fresh,
empty collection.

Usage::

    python -m src.scripts.reset_chroma
"""

import logging
import shutil
from pathlib import Path

import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

CHROMA_PATH: Path = Path("data/chroma")
COLLECTION_NAME: str = "fashion_knowledge"


def main() -> None:
    """Wipes the ChromaDB directory and recreates the collection."""
    # Fully remove the directory on disk to guarantee a clean state
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        logger.info("Removed ChromaDB directory: %s", CHROMA_PATH)
    else:
        logger.info("ChromaDB directory does not exist: %s — nothing to remove.", CHROMA_PATH)

    # Recreate from scratch
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    client.create_collection(name=COLLECTION_NAME)
    logger.info("Created fresh collection '%s'.", COLLECTION_NAME)

    print("ChromaDB reset successfully")


if __name__ == "__main__":
    main()

