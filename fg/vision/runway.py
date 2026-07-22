"""Runway visual grounding — the multimodal-KG bridge.

Builds a visual index over labeled runway imagery (``data/raw/vogue_runway/``,
~2.2k images across ~11 houses, each with designer/collection/season labels),
then links a user's look to the *nearest real runway looks* by image↔image
similarity — and traverses the KG from the matched designers for lineage.

This replaces Path A's lossy image↔text matching with image↔image against actual
collections, so "reads minimalist" becomes "nearest to Marni FW-26 / Rick Owens".
Also the substrate for per-node visual centroids (the multimodal KG node).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fg.config import settings
from fg.vision.index import VisualIndex

logger: logging.Logger = logging.getLogger(__name__)


def _default_runway_index_path() -> Path:
    return settings.embeddings_dir / settings.runway_index_name


def build_runway_index(
    embedder: Any,
    source_root: str | Path | None = None,
    out_path: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 64,
) -> Path:
    """Embeds labeled runway images into a ``VisualIndex``.

    Reads each look's JSON sidecar (``designer``, ``show``, ``season``,
    ``type``), loads the matching image, embeds with the fashion embedder, and
    saves an ``.npz`` index.

    Args:
        embedder: A ready ``FashionEmbedder``.
        source_root: Runway dir; defaults to ``data/raw/vogue_runway``.
        out_path: Index path; defaults to the configured runway index.
        limit: Optional cap on images.
        batch_size: Embedding batch size.

    Returns:
        Path of the written index.

    Raises:
        FileNotFoundError: If no runway JSON sidecars are found.
        RuntimeError: If Pillow is unavailable.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Need Pillow: pip install Pillow") from exc

    root = Path(source_root) if source_root else settings.data_dir / "raw" / "vogue_runway"
    jsons = sorted(root.rglob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No runway JSON sidecars under {root}.")

    vecs: list[np.ndarray] = []
    metas: list[dict] = []
    buf_imgs: list[Any] = []
    buf_meta: list[dict] = []
    seen = 0

    def _flush() -> None:
        if buf_imgs:
            vecs.append(embedder.encode_images(buf_imgs, batch_size=batch_size))
            metas.extend(buf_meta)
            buf_imgs.clear()
            buf_meta.clear()

    for jp in jsons:
        try:
            info = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        img_path = jp.with_suffix(".png")
        if not img_path.exists():
            # Fall back to the path recorded in the JSON.
            rec = info.get("image_path")
            img_path = (settings.repo_root / rec) if rec else img_path
        if not img_path.exists():
            continue
        try:
            buf_imgs.append(Image.open(img_path).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
        buf_meta.append({
            "source": "runway",
            "designer": info.get("designer", ""),
            "show": info.get("show", ""),
            "season": info.get("season", ""),
            "type": info.get("type", ""),
            "title": f"{info.get('designer','')} — {info.get('show','')}",
        })
        seen += 1
        if len(buf_imgs) >= batch_size:
            _flush()
            logger.info("Embedded %d runway looks…", seen)
        if limit is not None and seen >= limit:
            break
    _flush()

    if not vecs:
        raise RuntimeError("No runway images embedded.")
    index = VisualIndex(np.concatenate(vecs, axis=0), metas)
    return index.save(out_path or _default_runway_index_path())


class RunwayLinker:
    """Links a look to nearest runway designers/collections (image↔image).

    Attributes:
        index: The runway ``VisualIndex``.
    """

    def __init__(self, index_path: str | Path | None = None) -> None:
        """Loads the runway visual index.

        Args:
            index_path: Path to the runway ``.npz``; defaults to config.

        Raises:
            FileNotFoundError: If the index is missing.
        """
        self.index = VisualIndex.load(index_path or _default_runway_index_path())

    def link(self, look_vec: np.ndarray, top_k: int = 12, n_designers: int = 3) -> dict:
        """Finds nearest runway looks and aggregates designers/collections.

        Args:
            look_vec: The user look's image embedding.
            top_k: Neighbours to retrieve and vote over.
            n_designers: How many aggregated designers to return.

        Returns:
            ``{"designers": [(name, score)], "collections": [(show, score)],
            "nearest": [hit, …]}``.
        """
        hits = self.index.search(look_vec, top_k=top_k)
        by_designer: dict[str, float] = defaultdict(float)
        by_collection: dict[str, float] = defaultdict(float)
        for h in hits:
            d = h.get("designer") or "unknown"
            by_designer[d] += h["score"]
            show = h.get("title") or h.get("show") or d
            by_collection[show] += h["score"]
        designers = sorted(by_designer.items(), key=lambda kv: -kv[1])[:n_designers]
        collections = sorted(by_collection.items(), key=lambda kv: -kv[1])[:n_designers]
        return {
            "designers": [(d, round(s, 3)) for d, s in designers],
            "collections": [(c, round(s, 3)) for c, s in collections],
            "nearest": hits[:5],
        }
