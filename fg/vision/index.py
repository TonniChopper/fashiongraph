"""Visual index — L2-normalised image embeddings + parallel metadata.

Cosine search is pure ``numpy`` (no torch needed to *query* the index), so it
is trivially testable and cheap to load in the Personal Stylist path. Building
the index (embedding images) needs the ``FashionEmbedder`` and is a one-off.

On-disk format: a single ``.npz`` holding ``embeddings`` (float32 ``(N, D)``)
and a JSON string of the metadata list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fg.config import settings

if TYPE_CHECKING:  # pragma: no cover
    from fg.vision.embedder import FashionEmbedder

logger: logging.Logger = logging.getLogger(__name__)


def _default_index_path() -> Path:
    """Returns the default visual-index path from config."""
    name = settings.visual_index_name
    if name.endswith(".pt"):
        name = name[:-3] + ".npz"
    return settings.embeddings_dir / name


class VisualIndex:
    """In-memory visual index with numpy cosine search.

    Attributes:
        embeddings: L2-normalised embeddings, shape ``(N, D)``.
        metadata: Parallel list of metadata dicts, length ``N``.
    """

    def __init__(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Initializes and normalizes the index.

        Args:
            embeddings: ``(N, D)`` float array.
            metadata: Length-``N`` list of dicts.

        Raises:
            ValueError: If counts mismatch.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"metadata ({len(metadata)}) != embeddings ({embeddings.shape[0]})"
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.clip(norms, 1e-8, None)
        self.metadata = metadata

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def search(self, query: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Returns the top-k most similar items to *query*.

        Args:
            query: A single embedding, shape ``(D,)`` or ``(1, D)``.
            top_k: Number of results.

        Returns:
            Result dicts with ``rank``, ``score`` (cosine), ``index``, and all
            metadata keys, sorted by descending score.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = self.embeddings @ q  # (N,)
        k = min(top_k, sims.shape[0])
        top = np.argpartition(-sims, range(k))[:k]
        top = top[np.argsort(-sims[top])]
        return [
            {"rank": r, "score": round(float(sims[i]), 4), "index": int(i),
             **self.metadata[int(i)]}
            for r, i in enumerate(top, start=1)
        ]

    def save(self, path: str | Path | None = None) -> Path:
        """Saves the index to a ``.npz`` file.

        Args:
            path: Destination; defaults to the configured index path.

        Returns:
            The path written.
        """
        p = Path(path) if path else _default_index_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, embeddings=self.embeddings,
                 metadata=np.array(json.dumps(self.metadata)))
        logger.info("Saved visual index → %s (%d items)", p, len(self))
        return p

    @classmethod
    def load(cls, path: str | Path | None = None) -> "VisualIndex":
        """Loads an index from a ``.npz`` file.

        Args:
            path: Source; defaults to the configured index path.

        Returns:
            The loaded index.

        Raises:
            FileNotFoundError: If the file is missing.
        """
        p = Path(path) if path else _default_index_path()
        if not p.exists():
            raise FileNotFoundError(
                f"Visual index not found: {p}. Build it with `fgraph vision build`."
            )
        data = np.load(p, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        return cls(data["embeddings"], metadata)


def build_product_index(
    embedder: "FashionEmbedder",
    source_root: str | Path | None = None,
    out_path: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 64,
) -> Path:
    """Builds a visual index from the product-image dataset.

    Streams product rows, decodes images, embeds them in batches, and saves an
    ``.npz`` index. Designed to run on the M4 (MPS) — this is the heavy one-off.

    Args:
        embedder: A ready ``FashionEmbedder``.
        source_root: Dir with the product parquet(s); defaults to
            ``data/raw/fashion-product-images-small``.
        out_path: Where to write the index; defaults to the configured path.
        limit: Optional cap on number of images (for a quick first build).
        batch_size: Embedding batch size.

    Returns:
        The path of the written index.

    Raises:
        FileNotFoundError: If no parquet files are found.
        RuntimeError: If pandas/Pillow are unavailable.
    """
    import io

    try:
        import pandas as pd
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Need pandas + Pillow: pip install pandas pyarrow Pillow") from exc

    root = Path(source_root) if source_root else settings.data_dir / "raw" / "fashion-product-images-small"
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {root}. Download the dataset first.")

    def _decode(cell: Any):
        if isinstance(cell, dict) and cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"]))
        if isinstance(cell, (bytes, bytearray)):
            return Image.open(io.BytesIO(cell))
        return None

    all_vecs: list[np.ndarray] = []
    all_meta: list[dict] = []
    buf_imgs: list[Any] = []
    buf_meta: list[dict] = []
    seen = 0

    def _flush() -> None:
        if not buf_imgs:
            return
        vecs = embedder.encode_images(buf_imgs, batch_size=batch_size)
        all_vecs.append(vecs)
        all_meta.extend(buf_meta)
        buf_imgs.clear()
        buf_meta.clear()

    for fp in files:
        df = pd.read_parquet(fp)
        img_col = next((c for c in ("image", "img", "picture") if c in df.columns), None)
        if img_col is None:
            logger.warning("No image column in %s — skipping.", fp)
            continue
        for row in df.itertuples(index=False):
            rowd = row._asdict()
            img = _decode(rowd.get(img_col))
            if img is None:
                continue
            buf_imgs.append(img)
            buf_meta.append(_product_meta(rowd))
            seen += 1
            if len(buf_imgs) >= batch_size:
                _flush()
                logger.info("Embedded %d images…", seen)
            if limit is not None and seen >= limit:
                break
        if limit is not None and seen >= limit:
            break
    _flush()

    if not all_vecs:
        raise RuntimeError("No images embedded — check the dataset.")
    embeddings = np.concatenate(all_vecs, axis=0)
    index = VisualIndex(embeddings, all_meta)
    return index.save(out_path)


def _product_meta(row: dict) -> dict:
    """Extracts compact metadata for a product row."""
    keep = ("productDisplayName", "gender", "masterCategory", "subCategory",
            "articleType", "baseColour", "season", "year", "usage")
    meta = {"source": "fashion_products"}
    for k in keep:
        if k in row and row[k] is not None:
            meta[{"productDisplayName": "title", "baseColour": "colour",
                  "articleType": "category"}.get(k, k)] = str(row[k])
    return meta
