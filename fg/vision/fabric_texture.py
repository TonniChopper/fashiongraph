"""Fabric-texture visual grounding — mirror nodes for material entities.

Turns a folder of fabric-texture images (labeled by fabric name) into a visual
index, so a close-up swatch or garment crop can be matched to a fabric
(image↔image), and each fabric gets a **visual prototype** (a mean embedding)
attached to its KG ``material`` node — the "mirror node" from MMKG work.

Dataset-agnostic: works on any ``root/<fabric>/*.jpg`` (folder-per-fabric)
layout — DTD, Ten Fabrics, or a text2fabric export reorganized that way. The
fabric label comes from the parent folder name (normalised to the KG's
canonical material key).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fg.config import settings
from fg.kg.schema import normalize_entity
from fg.vision.index import VisualIndex

logger: logging.Logger = logging.getLogger(__name__)

_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def _default_texture_index_path() -> Path:
    return settings.embeddings_dir / settings.texture_index_name


def build_texture_index(
    embedder: Any,
    root: str | Path,
    out_path: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 64,
) -> Path:
    """Embeds a folder-per-fabric image set into a ``VisualIndex``.

    Args:
        embedder: A ready ``FashionEmbedder``.
        root: Directory laid out ``root/<fabric>/*.jpg``.
        out_path: Index path; defaults to the configured texture index.
        limit: Optional cap on images.
        batch_size: Embedding batch size.

    Returns:
        Path of the written index.

    Raises:
        FileNotFoundError: If no images are found.
        RuntimeError: If Pillow is unavailable.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Need Pillow: pip install Pillow") from exc

    root = Path(root)
    images: list[Path] = []
    for ext in _IMG_EXTS:
        images.extend(root.rglob(ext))
    images = sorted(images)
    if not images:
        raise FileNotFoundError(f"No fabric images under {root}.")

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

    for img_path in images:
        # Fabric label = the parent folder (skip if image sits directly in root).
        label = img_path.parent.name if img_path.parent != root else img_path.stem
        try:
            buf_imgs.append(Image.open(img_path).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
        buf_meta.append({
            "source": "fabric_texture",
            "fabric": normalize_entity(label),
            "title": label,
        })
        seen += 1
        if len(buf_imgs) >= batch_size:
            _flush()
            logger.info("Embedded %d fabric swatches…", seen)
        if limit is not None and seen >= limit:
            break
    _flush()

    if not vecs:
        raise RuntimeError("No fabric images embedded.")
    index = VisualIndex(np.concatenate(vecs, axis=0), metas)
    return index.save(out_path or _default_texture_index_path())


class FabricTextureLinker:
    """Identifies fabrics from a crop and exposes per-fabric visual centroids.

    Attributes:
        index: The fabric-texture ``VisualIndex``.
    """

    def __init__(self, index_path: str | Path | None = None) -> None:
        """Loads the fabric-texture index.

        Args:
            index_path: Path to the texture ``.npz``; defaults to config.

        Raises:
            FileNotFoundError: If the index is missing.
        """
        self.index = VisualIndex.load(index_path or _default_texture_index_path())

    def identify(self, crop_vec: np.ndarray, top_k: int = 8, n_fabrics: int = 3) -> list[tuple[str, float]]:
        """Returns the most likely fabrics for a swatch/garment-crop embedding.

        Args:
            crop_vec: An image embedding of a close-up crop.
            top_k: Neighbours to vote over.
            n_fabrics: How many aggregated fabrics to return.

        Returns:
            ``(fabric, score)`` pairs, best first.
        """
        hits = self.index.search(crop_vec, top_k=top_k)
        agg: dict[str, float] = defaultdict(float)
        for h in hits:
            agg[h.get("fabric", "?")] += h["score"]
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:n_fabrics]
        return [(f, round(s, 3)) for f, s in ranked]

    def centroids(self) -> dict[str, np.ndarray]:
        """Computes a mean (L2-normalised) embedding per fabric — the mirror node.

        Returns:
            ``fabric_key -> centroid embedding``.
        """
        groups: dict[str, list[np.ndarray]] = defaultdict(list)
        for i, meta in enumerate(self.index.metadata):
            groups[meta.get("fabric", "?")].append(self.index.embeddings[i])
        out: dict[str, np.ndarray] = {}
        for fabric, vs in groups.items():
            c = np.mean(np.stack(vs), axis=0)
            out[fabric] = c / max(float(np.linalg.norm(c)), 1e-8)
        return out
