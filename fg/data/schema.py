"""Normalized document schema for the ingest pipeline.

Every source is converted into ``Document`` objects with a text body and a
provenance-carrying metadata dict, so the knowledge core stays traceable:
each chunk can always answer "where did this come from?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Metadata keys we try to populate on every document (any may be absent).
CANONICAL_META_KEYS: tuple[str, ...] = (
    "source",       # dataset / origin id, e.g. "wikipedia", "fashion_products"
    "source_type",  # "encyclopedia" | "catalog" | "review" | "editorial" | ...
    "title",        # human-readable title / product name
    "url",          # link back to origin, if any
    "date",         # ISO date string, if known
    "brand",
    "category",
    "season",
    "year",
)


@dataclass
class Document:
    """A single normalized document ready for cleaning + indexing.

    Attributes:
        text: The document body (plain text).
        metadata: Provenance + attributes; ``source`` should always be set.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"

    @property
    def source(self) -> str:
        """The document's source id."""
        return str(self.metadata.get("source", "unknown"))
