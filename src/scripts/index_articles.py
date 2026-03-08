"""Process clean article dicts into ChromaDB with HuggingFace embeddings.

Takes a list of article dictionaries (``title``, ``clean_text``, ``date``),
splits the text into chunks with ``RecursiveCharacterTextSplitter``, embeds
each chunk via ``sentence-transformers/all-MiniLM-L6-v2``, and inserts
them into a persistent ChromaDB collection named ``fashion_knowledge``.

No LLMs are used — only exact splitting and vector insertion.

Usage::

    python -m src.scripts.index_articles
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

CHROMA_DIR: str = "data/chroma"
COLLECTION_NAME: str = "fashion_knowledge"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150


# ------------------------------------------------------------------
# Core function
# ------------------------------------------------------------------


def index_articles(articles: list[dict[str, str]]) -> int:
    """Splits and indexes a list of article dicts into ChromaDB.

    Each article dict must contain at least ``clean_text``.  Optional
    keys ``title`` and ``date`` are carried through as chunk metadata.

    Args:
        articles: List of dicts with keys ``title``, ``clean_text``,
            and ``date``.

    Returns:
        Total number of chunks inserted.
    """
    # ---- text splitter ----------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    # ---- embedding model --------------------------------------------
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)

    # ---- ChromaDB ---------------------------------------------------
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready (existing: %d docs).",
        COLLECTION_NAME, collection.count(),
    )

    # ---- split + embed + insert -------------------------------------
    total_chunks: int = 0

    for idx, article in enumerate(articles):
        text: str = article.get("clean_text", "")
        if not text.strip():
            continue

        title: str = article.get("title", "untitled")
        date: str = article.get("date", "unknown")
        source: str = article.get("source", "unknown")

        chunks: list[str] = splitter.split_text(text)
        if not chunks:
            continue

        # Deterministic IDs so re-runs are idempotent
        ids: list[str] = [
            hashlib.md5(
                f"{title}_{i}_{chunk[:80]}".encode()
            ).hexdigest()
            for i, chunk in enumerate(chunks)
        ]

        metadatas: list[dict[str, str]] = [
            {
                "title": title,
                "date": date,
                "source": source,
                "chunk_index": str(i),
            }
            for i in range(len(chunks))
        ]

        # Compute embeddings for the batch
        vectors: list[list[float]] = embeddings.embed_documents(chunks)

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=vectors,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

        if (idx + 1) % 10 == 0:
            logger.info(
                "Progress: %d / %d articles (%d chunks so far).",
                idx + 1, len(articles), total_chunks,
            )

    logger.info(
        "✅ Indexed %d chunks from %d articles. Collection size: %d.",
        total_chunks, len(articles), collection.count(),
    )
    return total_chunks


# ------------------------------------------------------------------
# CLI: load articles from expert_pairs or a JSON file
# ------------------------------------------------------------------

ARTICLES_DIR: Path = Path("data/raw")


def _load_articles_from_raw() -> list[dict[str, str]]:
    """Builds article dicts from .txt files under data/raw/.

    Handles two formats:
    - **Wikipedia files**: have ``TITLE:`` / ``SOURCE:`` / ``TYPE:``
      header lines at the top.
    - **Trafilatura files**: plain text with the title inferred from
      the filename.

    Returns:
        List of article dicts.
    """
    articles: list[dict[str, str]] = []
    for path in sorted(ARTICLES_DIR.rglob("*.txt")):
        try:
            text: str = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not text or len(text.split()) < 50:
            continue

        title: str = path.stem.replace("-", " ").replace("_", " ").title()
        source: str = "unknown"
        date: str = "unknown"

        # Parse Wikipedia-style headers (TITLE: / SOURCE: / TYPE:)
        if text.startswith("TITLE:"):
            header_lines: list[str] = []
            body_lines: list[str] = text.split("\n")
            for i, line in enumerate(body_lines):
                if line.startswith("TITLE:"):
                    title = line.split(":", 1)[1].strip()
                elif line.startswith("SOURCE:"):
                    source = line.split(":", 1)[1].strip()
                elif line.startswith("TYPE:"):
                    pass  # just skip
                elif line.strip() == "":
                    text = "\n".join(body_lines[i + 1:]).strip()
                    break

        # Detect source from parent directory
        parent: str = path.parent.name
        if parent == "wikipedia":
            source = f"wikipedia:{path.stem}"
        elif parent == "trafilatura":
            source = f"article:{path.stem}"

        articles.append({
            "title": title,
            "clean_text": text,
            "date": date,
            "source": source,
        })
    return articles


def main() -> None:
    """Entry point: loads articles from data/raw/ and indexes them."""
    articles: list[dict[str, str]] = _load_articles_from_raw()
    logger.info("Loaded %d articles from %s.", len(articles), ARTICLES_DIR)

    if not articles:
        logger.warning("No articles to index. Exiting.")
        return

    index_articles(articles)


if __name__ == "__main__":
    main()

