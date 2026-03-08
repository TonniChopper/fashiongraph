"""Convert runway_clip.pt to fashion_index.pt + fashion_meta.json.

Splits the combined output from ``embed_runway.py`` into the two files
expected by ``VisualFashionRetriever``:

- ``data/embeddings/fashion_index.pt`` — L2-normalised tensor ``(N, 512)``
- ``data/embeddings/fashion_meta.json`` — list of ``N`` metadata dicts

Usage::

    python -m src.scripts.build_fashion_index
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

SOURCE: Path = Path("data/embeddings/runway_clip.pt")
INDEX_OUT: Path = Path("data/embeddings/fashion_index.pt")
META_OUT: Path = Path("data/embeddings/fashion_meta.json")


def main() -> None:
    """Splits runway_clip.pt into fashion_index.pt + fashion_meta.json."""
    if not SOURCE.exists():
        logger.error("Source file not found: %s", SOURCE)
        logger.info("Run  python -m src.scripts.embed_runway  first.")
        return

    data = torch.load(SOURCE, map_location="cpu", weights_only=False)

    embeddings: torch.Tensor = data["embeddings"]
    metadata: list[dict[str, str]] = data["metadata"]

    # Ensure L2-normalised
    norms = embeddings.norm(dim=-1, keepdim=True)
    embeddings = embeddings / norms.clamp(min=1e-8)

    torch.save(embeddings, INDEX_OUT)
    logger.info("Saved index: %s  shape=%s", INDEX_OUT, tuple(embeddings.shape))

    with open(META_OUT, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    logger.info("Saved metadata: %s  entries=%d", META_OUT, len(metadata))

    logger.info("✅ Done. Use with VisualFashionRetriever.")


if __name__ == "__main__":
    main()

