"""Cross-domain aesthetic lineage — read a look through art & architecture.

Fashion shares compositional DNA with art and architecture (proportion, rhythm,
structure, ornament vs. reduction). Because the fashion embedder (SigLIP) shares
an image/text space, we can match a look *image* directly against a curated
lexicon of movement *text* prototypes — no art dataset download, no fragile
image-to-image mapping.

This is enrichment, not judgment: it gives the stylist evocative references
("this reads Bauhaus — reductive, functional, tonal"), not a verdict. Treat the
matches as inspiration prompts, and validate before trusting them.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

#: Curated art/architecture/design movements with fashion-relevant visual
#: descriptors. Descriptions are written to describe *visual qualities* so they
#: match well in an image-text embedding space.
MOVEMENTS: dict[str, str] = {
    "Minimalism": "reductive, clean lines, empty space, monochrome, unadorned, essential",
    "Bauhaus": "geometric, functional, primary structure, industrial, balanced, tonal",
    "De Stijl": "grid, primary colours, rectilinear, flat planes, strict geometry",
    "Constructivism": "structural, angular, utilitarian, bold diagonal, industrial red and black",
    "Brutalism": "raw, monolithic, heavy, concrete grey, unpolished, imposing volume",
    "Art Deco": "streamlined, symmetrical, luxe, metallic, geometric ornament, elongated",
    "Baroque": "ornate, dramatic, maximal, rich texture, gilded, theatrical contrast",
    "Rococo": "delicate, pastel, decorative, playful, ornamental, soft curves",
    "Gothic": "dark, vertical, pointed, dramatic silhouette, austere, sculptural",
    "Romanticism": "flowing, emotional, natural, soft draping, moody, painterly",
    "Art Nouveau": "organic, sinuous curves, botanical motifs, flowing lines, ornamental",
    "Surrealism": "unexpected, dreamlike, juxtaposition, subversive, uncanny detail",
    "Cubism": "fragmented, angular planes, abstract geometry, deconstructed form",
    "Futurism": "dynamic, sleek, speed lines, metallic, forward-leaning, technological",
    "Pop Art": "bold, saturated colour, graphic, playful, commercial, high contrast",
    "Op Art": "hypnotic pattern, black and white, optical illusion, repetition, contrast",
    "Memphis Design": "clashing colour, playful geometry, postmodern, irreverent, patterned",
    "Deconstructivism": "fragmented, asymmetric, exposed structure, raw seams, unresolved",
    "Wabi-Sabi": "imperfect, natural texture, muted earth tones, worn, humble, understated",
    "Streamline Moderne": "aerodynamic, smooth curves, horizontal, pale, polished, modern",
    "Functionalism": "practical, technical, utilitarian, no ornament, performance-driven",
    "Postmodernism": "eclectic, ironic, mixed references, historical pastiche, playful",
}


class MovementMatcher:
    """Matches a look embedding to art/architecture movement prototypes.

    Attributes:
        names: Movement names in prototype order.
        prototypes: L2-normalised text-embedding matrix ``(M, D)``.
    """

    def __init__(self, embedder: Any, movements: dict[str, str] | None = None) -> None:
        """Embeds the movement descriptors once with the fashion embedder.

        Args:
            embedder: A ``FashionEmbedder`` (must share space with look images).
            movements: Optional override of the movement lexicon.

        Raises:
            RuntimeError: If embedding the prototypes fails.
        """
        self._movements = movements or MOVEMENTS
        self.names = list(self._movements.keys())
        descriptions = [f"{n}: {d}" for n, d in self._movements.items()]
        protos = embedder.encode_texts(descriptions)
        protos = np.asarray(protos, dtype=np.float32)
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        self.prototypes = protos / np.clip(norms, 1e-8, None)
        logger.info("MovementMatcher ready: %d movements.", len(self.names))

    def match(self, look_vec: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """Returns the nearest movements to a look embedding.

        Args:
            look_vec: A look image embedding, shape ``(D,)`` or ``(1, D)``.
            top_k: Number of movements to return.

        Returns:
            ``(movement_name, similarity)`` pairs, best first.
        """
        q = np.asarray(look_vec, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = self.prototypes @ q
        k = min(top_k, sims.shape[0])
        order = np.argsort(-sims)[:k]
        return [(self.names[i], round(float(sims[i]), 4)) for i in order]
