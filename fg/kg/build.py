"""Build the fashion KG from the existing corpus (narrow slice first).

Iterates knowledge documents (Wikipedia fashion pages by default — brand and
designer articles, ideal KG fuel), runs LLM extraction, and stores triples.
Start with ``--limit 15`` (a handful of brands) to validate the slice before
scaling, exactly as the council advised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fg.kg.extractor import extract_triples
from fg.kg.store import KnowledgeGraph
from fg.llm.base import LLM

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BuildStats:
    """KG build counters.

    Attributes:
        docs: Documents processed.
        triples_added: New triples inserted.
        per_doc: Triple counts keyed by document title.
    """

    docs: int = 0
    triples_added: int = 0
    per_doc: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Plain-dict summary."""
        return {"docs": self.docs, "triples_added": self.triples_added,
                "per_doc": self.per_doc}


def _chunks(text: str, size: int = 2500, overlap: int = 200, max_chunks: int = 4) -> list[str]:
    """Splits *text* into overlapping windows (whole-article coverage).

    Args:
        text: Document text.
        size: Window size in characters.
        overlap: Overlap between consecutive windows.
        max_chunks: Cap on windows per document (keeps builds bounded).

    Returns:
        Up to ``max_chunks`` text windows.
    """
    out: list[str] = []
    start = 0
    while start < len(text) and len(out) < max_chunks:
        out.append(text[start:start + size])
        start += size - overlap
    return out


def build_kg(
    llm: LLM,
    source: str = "wikipedia",
    limit: int | None = 15,
    db_path: str | Path | None = None,
    chunks_per_doc: int = 4,
) -> BuildStats:
    """Builds/updates the KG from a corpus source.

    Extracts from multiple windows across each document (not just the intro),
    which materially increases triple yield and diversity.

    Args:
        llm: LLM backend for extraction.
        source: Registered ingest source name (default ``"wikipedia"``).
        limit: Max documents to process (narrow slice first).
        db_path: KG database path; defaults to config path.
        chunks_per_doc: Windows to extract per document.

    Returns:
        A :class:`BuildStats` summary.
    """
    from fg.data.sources import get_source

    spec = get_source(source)
    kg = KnowledgeGraph(db_path)
    stats = BuildStats()

    for doc in spec.loader(spec.default_root()):
        title = str(doc.metadata.get("title", "doc"))
        added = 0
        for chunk in _chunks(doc.text, max_chunks=chunks_per_doc):
            triples = extract_triples(chunk, llm, source=title)
            added += kg.add_triples(triples)
        stats.docs += 1
        stats.triples_added += added
        stats.per_doc[title] = added
        logger.info("KG: '%s' → +%d triples (total %d)", title, added, kg.count())
        if limit is not None and stats.docs >= limit:
            break

    logger.info("KG build complete: %s", stats.as_dict())
    kg.close()
    return stats
