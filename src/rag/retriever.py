"""Semantic retriever for fashion knowledge base."""

import logging

import chromadb

logger: logging.Logger = logging.getLogger(__name__)


class FashionRetriever:
    """Retrieves relevant fashion context for a given query.

    Used by Fashion LLM to ground responses in real fashion knowledge
    rather than hallucinating.
    """

    def __init__(self, persist_dir: str = "data/chroma") -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="fashion_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, n_results: int = 5,
                 filters: dict | None = None) -> list[dict]:
        """Retrieves top-n relevant chunks for a query.

        Args:
            query: Natural language query (e.g. "wide-leg jeans trend 2024").
            n_results: Number of chunks to return.
            filters: Optional ChromaDB metadata filters.

        Returns:
            List of dicts with keys: document, metadata, distance.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append({"document": doc, "metadata": meta, "distance": dist})

        logger.info("Retrieved %d chunks for query: '%s'", len(output), query)
        return output
