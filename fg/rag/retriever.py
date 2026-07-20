"""Semantic retriever for the fashion knowledge base.

Grounds LLM responses in real fashion knowledge instead of hallucinating.
Change from the old version: guards against an empty / missing collection so
callers get an empty list rather than a KeyError.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from fg.config import settings
from fg.rag.embeddings import get_text_embedding_function

logger: logging.Logger = logging.getLogger(__name__)


class FashionRetriever:
    """Retrieves relevant fashion context for a query.

    Attributes:
        client: Persistent ChromaDB client.
        collection: The fashion-knowledge collection.
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        """Initializes the retriever.

        Args:
            persist_dir: ChromaDB storage dir; defaults to
                ``settings.chroma_dir``.
        """
        persist = Path(persist_dir or settings.chroma_dir)
        self.client = chromadb.PersistentClient(path=str(persist))
        kwargs: dict = {
            "name": settings.chroma_collection,
            "metadata": {"hnsw:space": "cosine"},
        }
        embed_fn = get_text_embedding_function()
        if embed_fn is not None:
            kwargs["embedding_function"] = embed_fn
        self.collection = self.client.get_or_create_collection(**kwargs)

    def retrieve(
        self,
        query: str,
        n_results: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """Retrieves the top-n relevant chunks for *query*.

        Args:
            query: Natural-language query.
            n_results: Number of chunks; defaults to ``settings.rag_top_k``.
            filters: Optional ChromaDB metadata ``where`` filter.

        Returns:
            List of dicts with keys ``document``, ``metadata``, ``distance``.
            Empty if the collection has no documents.
        """
        n: int = n_results or settings.rag_top_k

        if self.collection.count() == 0:
            logger.warning("Retrieve called on empty collection — returning [].")
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            where=filters,
        )

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        output: list[dict] = [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        logger.info("Retrieved %d chunks for query: '%s'", len(output), query)
        return output
