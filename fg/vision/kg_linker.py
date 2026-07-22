"""Path A — look → knowledge-graph associative linking.

Because FashionSigLIP shares an image/text space and we have a fashion KG, we can
link an outfit *photo* to designers/brands/aesthetics without exact attribution:

1. Build a text descriptor for each KG entity FROM its graph facts
   ("Jil Sander — minimalism, tailoring, based in Milan").
2. Embed those descriptors → prototype matrix (the MovementMatcher pattern, but
   over KG entities).
3. Match the look's image embedding against the prototypes (shared space).
4. Surface the matched entities' KG facts (lineage, influences) for the stylist.

Result: *"reads minimalist — aligns with Jil Sander / The Row / Helmut Lang;
lineage traces to 1990s minimalism."* Inference-by-association, like a real
stylist — not attribution. Runtime match is pure numpy; only descriptor
embedding needs the model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

#: Relations that carry an entity's "design language", most salient first.
_DESCRIPTOR_RELATIONS: tuple[str, ...] = (
    "known_for", "has_silhouette", "uses_material", "associated_with",
    "from_era", "based_in", "influenced_by",
)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


class KGEntityLinker:
    """Links a look embedding to KG entities via text-descriptor prototypes.

    Attributes:
        names: Entity display names in prototype order.
        prototypes: L2-normalised text-embedding matrix ``(M, D)``.
    """

    def __init__(
        self,
        embedder: Any,
        kg: Any,
        max_entities: int = 400,
        min_facts: int = 3,
    ) -> None:
        """Builds descriptor prototypes from the KG's richest entities.

        Args:
            embedder: A ``FashionEmbedder`` (shares space with look images).
            kg: A ``KnowledgeGraph``.
            max_entities: Cap on entities to embed (by fact count).
            min_facts: Minimum facts for an entity to be a prototype.
        """
        self.kg = kg
        self.names: list[str] = []
        descriptors: list[str] = []
        for display, _key, count in kg.top_subjects(max_entities):
            if count < min_facts:
                continue
            desc = self._descriptor(display)
            if desc:
                self.names.append(display)
                descriptors.append(desc)

        if descriptors:
            protos = np.asarray(embedder.encode_texts(descriptors), dtype=np.float32)
            norms = np.linalg.norm(protos, axis=1, keepdims=True)
            self.prototypes = protos / np.clip(norms, 1e-8, None)
        else:
            self.prototypes = np.empty((0, 0), dtype=np.float32)
        logger.info("KGEntityLinker ready: %d entity prototypes.", len(self.names))

    def _descriptor(self, entity: str) -> str:
        """Builds a design-language descriptor for *entity* from its KG facts."""
        facts = self.kg.outgoing(entity)
        parts: list[str] = []
        for rel in _DESCRIPTOR_RELATIONS:
            objs = [f["object"] for f in facts if f["relation"] == rel][:3]
            if objs:
                parts.append(", ".join(objs))
        return f"{entity} — " + "; ".join(parts) if parts else ""

    def match(self, look_vec: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Returns the nearest KG entities to a look embedding.

        Args:
            look_vec: An image embedding ``(D,)`` or ``(1, D)``.
            top_k: Number of entities.

        Returns:
            ``(entity_name, similarity)`` pairs, best first (empty if no
            prototypes).
        """
        if self.prototypes.size == 0:
            return []
        q = _normalize(np.asarray(look_vec).reshape(-1))
        sims = self.prototypes @ q
        k = min(top_k, sims.shape[0])
        order = np.argsort(-sims)[:k]
        return [(self.names[i], round(float(sims[i]), 4)) for i in order]

    def link(self, look_vec: np.ndarray, top_k: int = 3, facts_per: int = 6) -> list[dict]:
        """Matches a look and attaches each entity's KG facts (its lineage).

        Args:
            look_vec: The look image embedding.
            top_k: How many entities to link.
            facts_per: Max KG facts to surface per entity.

        Returns:
            One dict per matched entity: ``{entity, score, facts}``.
        """
        out: list[dict] = []
        for name, score in self.match(look_vec, top_k=top_k):
            out.append({
                "entity": name,
                "score": score,
                "facts": self.kg.facts_as_text(name, limit=facts_per),
            })
        return out
