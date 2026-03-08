"""Visual fashion retriever using CLIP text-to-image similarity.

Loads a pre-built image embedding index (``fashion_index.pt``) and
corresponding metadata (``fashion_meta.json``), then retrieves the
most visually similar looks for a free-text query using the project's
``FashionCLIPEncoder``.

All embeddings must be L2-normalised so cosine similarity reduces to
a simple matrix multiplication.

Usage::

    python -m src.rag.visual_retriever "oversized leather coat dark minimal"
    python -m src.rag.visual_retriever --top-k 10 "red couture gown"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.models.clip_encoder import FashionCLIPEncoder

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Default paths (overridable via constructor)
# ------------------------------------------------------------------

DEFAULT_INDEX_PATH: Path = Path("data/embeddings/fashion_index.pt")
DEFAULT_META_PATH: Path = Path("data/embeddings/fashion_meta.json")
DEFAULT_CHECKPOINT: Path = Path("checkpoints/clip_epoch3.pt")


# ------------------------------------------------------------------
# Result dataclass-like dict
# ------------------------------------------------------------------


def _result_dict(
    rank: int,
    score: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Builds a single search-result dict.

    Args:
        rank: 1-based rank position.
        score: Cosine similarity score (0–1).
        metadata: Metadata dict for this look.

    Returns:
        Dict with ``rank``, ``score``, and all metadata keys.
    """
    return {"rank": rank, "score": round(score, 4), **metadata}


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class VisualFashionRetriever:
    """Retrieves runway looks by text query using CLIP embeddings.

    The retriever holds an in-memory matrix of L2-normalised image
    embeddings and a parallel list of metadata dicts.  Text queries
    are encoded by the same ``FashionCLIPEncoder`` that produced the
    image embeddings, so the shared embedding space allows direct
    cosine-similarity search.

    Attributes:
        model: The ``FashionCLIPEncoder`` used for text encoding.
        device: Torch device (``cuda`` or ``cpu``).
        embeddings: L2-normalised image embeddings, shape ``(N, D)``.
        metadata: List of metadata dicts, length ``N``.
    """

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        meta_path: str | Path = DEFAULT_META_PATH,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        embed_dim: int = 512,
        device: str | None = None,
    ) -> None:
        """Initializes the visual retriever.

        Args:
            index_path: Path to the ``.pt`` file containing L2-normalised
                image embeddings of shape ``(N, D)``.
            meta_path: Path to the ``.json`` file with a list of ``N``
                metadata dicts (keys: ``designer``, ``show``,
                ``image_path``, etc.).
            checkpoint_path: Optional path to a ``FashionCLIPEncoder``
                checkpoint.  If ``None`` or the file does not exist,
                pretrained weights are used.
            model_name: OpenCLIP architecture (must match the encoder that
                created the index).
            pretrained: OpenCLIP pretrained weight tag.
            embed_dim: Embedding dimensionality (must match the index).
            device: ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.

        Raises:
            FileNotFoundError: If *index_path* or *meta_path* do not exist.
            ValueError: If the embedding and metadata counts differ.
        """
        # ---- resolve paths ------------------------------------------
        index_path = Path(index_path)
        meta_path = Path(meta_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        # ---- device -------------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: torch.device = torch.device(device)

        # ---- load CLIP model ----------------------------------------
        self.model: FashionCLIPEncoder = FashionCLIPEncoder(
            model_name=model_name,
            pretrained=pretrained,
            embed_dim=embed_dim,
            freeze_backbone=True,
        )

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                state: dict[str, Any] = torch.load(
                    ckpt, map_location=self.device, weights_only=True,
                )
                if isinstance(state, dict) and "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                else:
                    self.model.load_state_dict(state)
                logger.info("Loaded encoder checkpoint: %s", ckpt)
            else:
                logger.info(
                    "Checkpoint %s not found — using pretrained weights.", ckpt,
                )

        self.model = self.model.to(self.device)
        self.model.eval()

        # ---- load index + metadata ----------------------------------
        raw = torch.load(index_path, map_location=self.device, weights_only=True)

        # Support two formats:
        #   1. Plain tensor (N, D)
        #   2. Dict with key "embeddings" (from embed_runway.py)
        if isinstance(raw, dict) and "embeddings" in raw:
            self.embeddings: torch.Tensor = raw["embeddings"].to(self.device)
        else:
            self.embeddings = raw.to(self.device)

        # Ensure float32 and L2-normalised
        self.embeddings = self.embeddings.float()
        norms: torch.Tensor = self.embeddings.norm(dim=-1, keepdim=True)
        self.embeddings = self.embeddings / norms.clamp(min=1e-8)

        with open(meta_path, encoding="utf-8") as fh:
            self.metadata: list[dict[str, Any]] = json.load(fh)

        if len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                f"Metadata length ({len(self.metadata)}) does not match "
                f"index size ({self.embeddings.shape[0]})"
            )

        logger.info(
            "VisualFashionRetriever ready: %d looks, dim=%d, device=%s",
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            self.device,
        )

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def search_looks(
        self,
        text_query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Finds the most similar looks for a text description.

        Tokenizes the query, encodes it through the CLIP text encoder
        and fashion projection head, L2-normalises the resulting
        embedding, and computes cosine similarities against the full
        image index via a single matrix multiplication.

        Args:
            text_query: Free-text fashion description (e.g.
                ``"oversized leather coat dark minimal"``).
            top_k: Number of results to return.

        Returns:
            A list of result dicts sorted by descending similarity.
            Each dict contains ``rank``, ``score``, and all metadata
            keys (``designer``, ``show``, ``image_path``, …).
        """
        # Encode query text
        txt_emb: torch.Tensor = self.model.encode_text([text_query])
        txt_emb = txt_emb.to(self.device).float()
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # Cosine similarity: (1, D) @ (D, N) → (1, N)
        similarities: torch.Tensor = torch.matmul(
            txt_emb, self.embeddings.T,
        ).squeeze(0)

        # Top-k
        top_k = min(top_k, similarities.shape[0])
        scores, indices = torch.topk(similarities, k=top_k)

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(
            zip(scores.cpu().tolist(), indices.cpu().tolist()), start=1,
        ):
            results.append(_result_dict(rank, score, self.metadata[idx]))

        logger.info(
            "search_looks(%r, top_k=%d) → %d results (best=%.4f)",
            text_query, top_k, len(results),
            results[0]["score"] if results else 0.0,
        )
        return results

    @torch.no_grad()
    def search_by_image(
        self,
        image: torch.Tensor,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Finds the most similar looks for an image embedding.

        Useful for "find similar looks" given a reference image.

        Args:
            image: Preprocessed image tensor of shape ``(1, C, H, W)``
                or ``(C, H, W)``.
            top_k: Number of results to return.

        Returns:
            A list of result dicts sorted by descending similarity.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        img_emb: torch.Tensor = self.model.encode_image(image)
        img_emb = img_emb.float()
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        similarities: torch.Tensor = torch.matmul(
            img_emb, self.embeddings.T,
        ).squeeze(0)

        top_k = min(top_k, similarities.shape[0])
        scores, indices = torch.topk(similarities, k=top_k)

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(
            zip(scores.cpu().tolist(), indices.cpu().tolist()), start=1,
        ):
            results.append(_result_dict(rank, score, self.metadata[idx]))

        logger.info(
            "search_by_image(top_k=%d) → %d results (best=%.4f)",
            top_k, len(results),
            results[0]["score"] if results else 0.0,
        )
        return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point for quick testing."""
    parser = argparse.ArgumentParser(
        description="Visual Fashion Retriever — search looks by text",
    )
    parser.add_argument(
        "query",
        type=str,
        help='Text query (e.g. "oversized leather coat dark minimal")',
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results (default: 5)",
    )
    parser.add_argument(
        "--index", type=Path, default=DEFAULT_INDEX_PATH,
        help="Path to fashion_index.pt",
    )
    parser.add_argument(
        "--meta", type=Path, default=DEFAULT_META_PATH,
        help="Path to fashion_meta.json",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Path to FashionCLIPEncoder checkpoint",
    )
    args = parser.parse_args()

    retriever = VisualFashionRetriever(
        index_path=args.index,
        meta_path=args.meta,
        checkpoint_path=args.checkpoint,
    )

    results: list[dict[str, Any]] = retriever.search_looks(
        args.query, top_k=args.top_k,
    )

    print(f"\n🔍 Query: \"{args.query}\"")
    print(f"   Top {len(results)} results:\n")
    for r in results:
        print(
            f"   #{r['rank']}  score={r['score']:.4f}  "
            f"designer={r.get('designer', '?')}  "
            f"show={r.get('show', '?')}"
        )
        if "image_path" in r:
            print(f"         {r['image_path']}")
    print()


if __name__ == "__main__":
    main()

