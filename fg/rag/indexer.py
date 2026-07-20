"""Fashion knowledge-base indexer for ChromaDB.

Takes cleaned text documents (curated editorial, brand/DNA facts, expert
annotations) and stores them as vector embeddings for semantic search.

Change from the old version: uses ``upsert`` instead of ``add`` so
re-indexing the same source is idempotent rather than raising on duplicate
IDs.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fg.config import settings
from fg.rag.embeddings import get_text_embedding_function

logger: logging.Logger = logging.getLogger(__name__)


class FashionKnowledgeIndexer:
    """Indexes fashion documents into ChromaDB for RAG retrieval.

    Attributes:
        client: Persistent ChromaDB client.
        collection: The fashion-knowledge collection (cosine space).
        splitter: Recursive character splitter for chunking.
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        """Initializes the indexer.

        Args:
            persist_dir: ChromaDB storage dir; defaults to
                ``settings.chroma_dir``.
        """
        persist = Path(persist_dir or settings.chroma_dir)
        persist.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist))
        kwargs: dict = {
            "name": settings.chroma_collection,
            "metadata": {"hnsw:space": "cosine"},
        }
        embed_fn = get_text_embedding_function()
        if embed_fn is not None:
            kwargs["embedding_function"] = embed_fn
        self.collection = self.client.get_or_create_collection(**kwargs)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        logger.info(
            "FashionKnowledgeIndexer ready, collection size: %d",
            self.collection.count(),
        )

    def add_document(self, text: str, metadata: dict) -> int:
        """Splits *text* into chunks and upserts them into ChromaDB.

        Args:
            text: Raw document text.
            metadata: Provenance dict — should include ``source`` and,
                where known, ``date``, ``type``, ``brand``, ``season``.

        Returns:
            Number of chunks upserted.

        Raises:
            ValueError: If *text* is empty after stripping.
        """
        if not text or not text.strip():
            raise ValueError("Cannot index empty document text.")

        chunks: list[str] = self.splitter.split_text(text)
        source: str = str(metadata.get("source", "doc"))
        ids: list[str] = [
            hashlib.md5(f"{source}_{i}_{chunk[:64]}".encode()).hexdigest()
            for i, chunk in enumerate(chunks)
        ]
        self.collection.upsert(
            documents=chunks,
            metadatas=[metadata] * len(chunks),
            ids=ids,
        )
        logger.info("Upserted %d chunks from source: %s", len(chunks), source)
        return len(chunks)

    def add_expert_annotation(
        self, element: str, season: str, year: int, context: str
    ) -> None:
        """Adds an expert trend annotation to the knowledge base.

        Args:
            element: Fashion element (e.g. ``"barrel-leg denim"``).
            season: Season label (e.g. ``"FW"``).
            year: Year of the annotation.
            context: Free-text analysis.
        """
        text: str = (
            f"Fashion trend analysis: {element} in {season} {year}. "
            f"Context: {context}"
        )
        self.add_document(
            text,
            metadata={
                "source": "expert_annotation",
                "element": element,
                "year": str(year),
                "season": season,
                "type": "trend_score",
            },
        )

    @property
    def size(self) -> int:
        """Number of chunks currently in the collection."""
        return self.collection.count()
