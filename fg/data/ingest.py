"""Ingest pipeline: source → clean → dedup → ChromaDB.

Orchestrates the loaders in ``fg.data.sources`` through the cleaning stage and
into the ``FashionKnowledgeIndexer``. This is the Phase-1 knowledge-core build.

    fgraph data list
    fgraph data build --source fashion_products,wikipedia --limit 2000
    fgraph data smoke "quiet luxury tailoring"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fg.data.clean import clean_documents
from fg.data.schema import Document
from fg.data.sources import SourceSpec, get_source

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    """Per-run ingest counters.

    Attributes:
        raw: Documents produced by loaders.
        kept: Documents surviving clean + dedup.
        chunks: Chunks written to the vector store.
        per_source: Raw counts keyed by source name.
    """

    raw: int = 0
    kept: int = 0
    chunks: int = 0
    per_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Returns a plain-dict summary."""
        return {
            "raw": self.raw,
            "kept": self.kept,
            "chunks": self.chunks,
            "per_source": self.per_source,
        }


def _load_source(spec: SourceSpec, root: Path | None, limit: int | None) -> list[Document]:
    """Loads (optionally truncated) documents from one source.

    Args:
        spec: The source spec.
        root: Override data directory; defaults to the spec's default.
        limit: Max documents to pull (before cleaning).

    Returns:
        Raw documents.
    """
    src_root = Path(root) if root else spec.default_root()
    docs: list[Document] = []
    for doc in spec.loader(src_root):
        docs.append(doc)
        if limit is not None and len(docs) >= limit:
            break
    logger.info("Source '%s': loaded %d raw docs from %s", spec.name, len(docs), src_root)
    return docs


def build(
    source_names: list[str],
    *,
    root: Path | None = None,
    limit: int | None = None,
    persist_dir: Path | None = None,
) -> IngestStats:
    """Builds / updates the knowledge index from the given sources.

    Args:
        source_names: Registry keys to ingest.
        root: Optional override for the raw-data directory (applies to all).
        limit: Optional per-source document cap (useful for smoke runs).
        persist_dir: Optional ChromaDB dir; defaults to ``settings.chroma_dir``.

    Returns:
        An :class:`IngestStats` summary.

    Raises:
        KeyError: If a source name is unknown.
    """
    # Import the indexer lazily so `fg data list` works without chromadb.
    from fg.rag.indexer import FashionKnowledgeIndexer

    indexer = FashionKnowledgeIndexer(persist_dir=persist_dir)
    stats = IngestStats()

    for name in source_names:
        spec = get_source(name)
        raw_docs = _load_source(spec, root, limit)
        stats.raw += len(raw_docs)
        stats.per_source[name] = len(raw_docs)

        cleaned = clean_documents(raw_docs)
        stats.kept += len(cleaned)
        logger.info("Source '%s': %d → %d after clean+dedup", name, len(raw_docs), len(cleaned))

        for doc in cleaned:
            stats.chunks += indexer.add_document(doc.text, doc.metadata)

    logger.info("Ingest complete: %s", stats.as_dict())
    return stats


def smoke(query: str, n_results: int = 5, persist_dir: Path | None = None) -> list[dict]:
    """Runs a retrieval smoke test against the built index.

    Args:
        query: Natural-language query.
        n_results: Number of chunks to return.
        persist_dir: Optional ChromaDB dir.

    Returns:
        Retrieved chunks (document, metadata, distance).
    """
    from fg.rag.retriever import FashionRetriever

    retriever = FashionRetriever(persist_dir=persist_dir)
    return retriever.retrieve(query, n_results=n_results)
