"""Perception stack — the composition root for the look-review pipeline.

Extracted from the CLI's ``_build_look_review`` (which the code graph flagged as
a 10-community bridge). Owns the load-and-degrade logic for every perception
component in one *testable* place, so new signals (VLM extraction, visual
centroids, cross-modal alignment) get a clean home instead of piling onto CLI
glue. Each component loads independently; a failure degrades to ``None`` with a
note rather than breaking the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger: logging.Logger = logging.getLogger(__name__)

Note = Callable[[str], None]


@dataclass
class PerceptionStack:
    """The assembled perception components (any may be ``None``).

    Attributes:
        embedder: Fashion image/text embedder.
        segmenter: Garment segmenter.
        visual_index: Product visual index (similar pieces).
        aesthetic_scorer: Learned taste scorer.
        movement_matcher: Art/architecture aesthetic-movement matcher.
        kg: Knowledge graph (also shared with the context builder).
        kg_linker: Text-descriptor KG entity linker.
        runway_linker: Runway image↔image designer linker.
    """

    embedder: Any | None = None
    segmenter: Any | None = None
    visual_index: Any | None = None
    aesthetic_scorer: Any | None = None
    movement_matcher: Any | None = None
    kg: Any | None = None
    kg_linker: Any | None = None
    runway_linker: Any | None = None


def _try(note: Note, name: str, fn: Callable[[], Any]) -> Any | None:
    """Runs a loader, returning its value or ``None`` (with a note) on failure."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        note(f"{name} unavailable: {exc}")
        return None


def build_perception_stack(
    embedder: Any | None = None,
    on_note: Note | None = None,
) -> PerceptionStack:
    """Loads all available perception components, degrading gracefully.

    Args:
        embedder: A pre-built ``FashionEmbedder`` to reuse; loaded if ``None``.
        on_note: Callback for "component unavailable" notes (defaults to logging).

    Returns:
        A :class:`PerceptionStack`; unavailable components are ``None``.
    """
    note: Note = on_note or logger.info
    s = PerceptionStack()

    if embedder is None:
        def _load_embedder():
            from fg.vision.embedder import FashionEmbedder
            return FashionEmbedder()
        embedder = _try(note, "fashion embedder", _load_embedder)
    s.embedder = embedder

    def _load_segmenter():
        from fg.vision.segmentation import GarmentSegmenter
        return GarmentSegmenter()
    s.segmenter = _try(note, "garment segmenter", _load_segmenter)

    def _load_product_index():
        from fg.vision.index import VisualIndex
        return VisualIndex.load()
    s.visual_index = _try(note, "product visual index", _load_product_index)

    def _load_scorer():
        from fg.vision.aesthetics import AestheticScorer
        return AestheticScorer.load()
    s.aesthetic_scorer = _try(note, "aesthetic scorer", _load_scorer)

    def _load_kg():
        from fg.kg.store import KnowledgeGraph, _default_db_path
        return KnowledgeGraph() if Path(_default_db_path()).exists() else None
    s.kg = _try(note, "knowledge graph", _load_kg)

    if s.embedder is not None:
        def _load_matcher():
            from fg.vision.aesthetic_movements import MovementMatcher
            return MovementMatcher(s.embedder)
        s.movement_matcher = _try(note, "movement matcher", _load_matcher)

        if s.kg is not None:
            def _load_kg_linker():
                from fg.vision.kg_linker import KGEntityLinker
                return KGEntityLinker(s.embedder, s.kg)
            s.kg_linker = _try(note, "kg linker", _load_kg_linker)

        def _load_runway():
            from fg.vision.runway import RunwayLinker, _default_runway_index_path
            return RunwayLinker() if Path(_default_runway_index_path()).exists() else None
        s.runway_linker = _try(note, "runway linker", _load_runway)

    return s
