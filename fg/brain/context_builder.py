"""Fusion Context builder.

Assembles the shared context every capability reasons over. Today it fuses RAG
knowledge + memory; the visual (CLIP), trend (GNN), and Brand-DNA signals plug
in here in later phases without changing capability code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class FusionContext:
    """Aggregated context passed to capabilities.

    Attributes:
        query: The retrieval query used to gather knowledge.
        rag_chunks: Retrieved knowledge chunks (document/metadata/distance).
        memory: Relevant memory snapshot (brand/user/session facts).
        signals: Reserved for later fusion inputs (visual, trend, dna).
    """

    query: str
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    kg_facts: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    def rag_text(self, max_chars: int = 2000) -> str:
        """Renders retrieved chunks as a compact, sourced context block.

        Args:
            max_chars: Soft cap on the rendered length.

        Returns:
            A newline-separated, source-tagged context string (may be empty).
        """
        lines: list[str] = []
        total = 0
        for chunk in self.rag_chunks:
            meta = chunk.get("metadata", {})
            tag = meta.get("title") or meta.get("source", "source")
            snippet = chunk.get("document", "").strip().replace("\n", " ")
            line = f"- [{tag}] {snippet}"
            total += len(line)
            if total > max_chars:
                break
            lines.append(line)
        return "\n".join(lines)

    def memory_text(self) -> str:
        """Renders memory as ``key: value`` lines (may be empty)."""
        return "\n".join(f"- {k}: {v}" for k, v in self.memory.items())

    def kg_text(self) -> str:
        """Renders knowledge-graph facts as bullet lines (may be empty)."""
        return "\n".join(f"- {fact}" for fact in self.kg_facts)

    def knowledge_block(self) -> str:
        """Combined grounding: structured KG facts first, then RAG passages.

        KG facts lead because they're precise, relational, and traceable —
        exactly the grounding a flat vector store can't give.
        """
        parts: list[str] = []
        if self.kg_facts:
            parts.append("Knowledge-graph facts:\n" + self.kg_text())
        rag = self.rag_text()
        if rag:
            parts.append("Retrieved passages:\n" + rag)
        return "\n\n".join(parts)


class ContextBuilder:
    """Builds :class:`FusionContext` objects from a retriever + memory.

    Attributes:
        retriever: Optional object exposing ``retrieve(query, n_results, filters)``.
    """

    def __init__(self, retriever: Any | None = None, kg: Any | None = None) -> None:
        """Initializes the builder.

        Args:
            retriever: A ``FashionRetriever``-like object, or ``None`` to run
                without RAG grounding.
            kg: A ``KnowledgeGraph``-like object exposing ``entities()`` and
                ``facts_as_text(entity)``, or ``None`` to run without KG facts.
        """
        self.retriever = retriever
        self.kg = kg
        self._kg_entities: set[str] | None = None

    def build(
        self,
        query: str,
        *,
        n_rag: int = 5,
        filters: dict | None = None,
        memory: dict | None = None,
    ) -> FusionContext:
        """Assembles a fusion context for *query*.

        Args:
            query: Retrieval query.
            n_rag: Number of knowledge chunks to fetch.
            filters: Optional metadata filter for retrieval.
            memory: Optional memory snapshot to attach.

        Returns:
            A populated :class:`FusionContext`. RAG failures degrade gracefully
            to an empty chunk list (capabilities still run, just ungrounded).
        """
        chunks: list[dict[str, Any]] = []
        if self.retriever is not None:
            try:
                chunks = self.retriever.retrieve(query, n_results=n_rag, filters=filters)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG retrieval failed (%s) — continuing ungrounded.", exc)

        kg_facts = self._kg_facts_for(query)
        return FusionContext(
            query=query, rag_chunks=chunks, kg_facts=kg_facts, memory=memory or {}
        )

    def _kg_facts_for(self, query: str, max_entities: int = 4, max_facts: int = 20) -> list[str]:
        """Finds KG entities mentioned in *query* and returns their facts.

        Args:
            query: The user query / retrieval string.
            max_entities: Cap on matched entities to expand.
            max_facts: Cap on total facts returned.

        Returns:
            De-duplicated fact strings (empty if no KG or no matches).
        """
        if self.kg is None:
            return []
        try:
            if self._kg_entities is None:
                self._kg_entities = self.kg.entities()
            q = query.lower()
            # Match multi-word entities first (more specific), then longer ones.
            matched = [e for e in self._kg_entities if e and e in q]
            matched.sort(key=len, reverse=True)
            facts: list[str] = []
            seen: set[str] = set()
            for ent in matched[:max_entities]:
                for fact in self.kg.facts_as_text(ent):
                    if fact not in seen:
                        seen.add(fact)
                        facts.append(fact)
                    if len(facts) >= max_facts:
                        return facts
            return facts
        except Exception as exc:  # noqa: BLE001
            logger.warning("KG lookup failed (%s).", exc)
            return []
