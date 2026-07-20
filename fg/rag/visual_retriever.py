"""Visual fashion retriever — search runway/catalog looks by text or image.

Loads a prebuilt L2-normalised image-embedding index + parallel metadata,
encodes queries with the project CLIP encoder, and ranks by cosine similarity.

Dual-path ranking: a text query is scored both by the CLIP text→image path
and by a sentence-transformer over metadata descriptors; the two ranked lists
are combined with **Reciprocal Rank Fusion** (fg.rag.fusion) instead of the
old hand-tuned ``0.8*clip + 0.2*text`` blend.

    python -m fg.rag.visual_retriever "oversized leather coat dark minimal"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from fg.config import settings
from fg.models.clip_encoder import FashionCLIPEncoder
from fg.rag.fusion import reciprocal_rank_fusion

logger: logging.Logger = logging.getLogger(__name__)


def _build_descriptor(meta: dict[str, Any]) -> str:
    """Builds a short text descriptor from metadata for text re-ranking.

    Args:
        meta: Metadata dict for one look.

    Returns:
        A descriptor string (possibly empty).
    """
    parts: list[str] = [
        str(meta[key])
        for key in ("designer", "show", "label")
        if meta.get(key)
    ]
    return " ".join(parts)


def _describe_hit(hit: dict[str, Any]) -> str:
    """Builds a human-readable one-line description from a result dict."""
    if hit.get("designer") or hit.get("show"):
        desc = f"{hit.get('designer', 'Unknown')} — {hit.get('show', 'Unknown show')}"
        if hit.get("look_index") is not None:
            desc += f", look {hit['look_index']}"
        return desc
    if hit.get("label"):
        return f"{hit['label']} (source: {hit.get('source', 'fashionpedia')})"
    return f"unlabeled look (source: {hit.get('source', 'unknown')})"


class VisualFashionRetriever:
    """Retrieves looks by text/image query using CLIP embeddings + RRF.

    Attributes:
        model: CLIP encoder used for query encoding.
        device: Torch device.
        embeddings: L2-normalised image embeddings ``(N, D)``.
        metadata: Parallel list of metadata dicts, length ``N``.
    """

    def __init__(
        self,
        combined_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        embed_dim: int = 512,
        device: str | None = None,
        text_rerank: bool = True,
    ) -> None:
        """Initializes the retriever.

        Args:
            combined_path: ``.pt`` file with ``{"embeddings", "metadata"}``;
                defaults to ``<embeddings_dir>/runway_clip.pt``.
            checkpoint_path: Optional encoder checkpoint; pretrained if missing.
            model_name: OpenCLIP architecture (must match the index).
            pretrained: OpenCLIP weight tag.
            embed_dim: Embedding dim (must match the index).
            device: ``"cuda"``/``"mps"``/``"cpu"``; auto-detected if ``None``.
            text_rerank: Whether to enable the sentence-transformer text path.

        Raises:
            FileNotFoundError: If the embedding file is missing.
            ValueError: If embeddings and metadata counts differ.
        """
        self.device = torch.device(device or self._auto_device())

        self.model = FashionCLIPEncoder(
            model_name=model_name,
            pretrained=pretrained,
            embed_dim=embed_dim,
            freeze_backbone=True,
        )
        self._maybe_load_checkpoint(checkpoint_path)
        self.model = self.model.to(self.device).eval()

        self._load_index(combined_path)
        self._normalise_embeddings()
        if len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                f"Metadata length ({len(self.metadata)}) != index size "
                f"({self.embeddings.shape[0]})"
            )
        logger.info(
            "VisualFashionRetriever ready: %d looks, dim=%d, device=%s",
            self.embeddings.shape[0], self.embeddings.shape[1], self.device,
        )

        self.text_model = None
        if text_rerank:
            try:
                from sentence_transformers import SentenceTransformer

                self.text_model = SentenceTransformer(
                    settings.text_embed_model, device=str(self.device)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Text re-rank disabled (%s).", exc)

    # ---- setup helpers ------------------------------------------------

    @staticmethod
    def _auto_device() -> str:
        """Picks cuda → mps → cpu."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _maybe_load_checkpoint(self, checkpoint_path: str | Path | None) -> None:
        """Loads an encoder checkpoint if one exists."""
        ckpt = Path(checkpoint_path or settings.checkpoints_dir / "clip_epoch3.pt")
        if not ckpt.exists():
            logger.info("No checkpoint at %s — using pretrained weights.", ckpt)
            return
        state = torch.load(ckpt, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        logger.info("Loaded encoder checkpoint: %s", ckpt)

    def _load_index(self, combined_path: str | Path | None) -> None:
        """Loads embeddings + metadata from a combined ``.pt`` file."""
        cp = Path(combined_path or settings.embeddings_dir / "runway_clip.pt")
        if not cp.exists():
            raise FileNotFoundError(f"Embedding index not found: {cp}")
        raw = torch.load(cp, map_location=self.device, weights_only=False)
        if not (isinstance(raw, dict) and "embeddings" in raw):
            raise ValueError(f"{cp} missing 'embeddings' key.")
        self.embeddings = raw["embeddings"].to(self.device).float()
        if "metadata" in raw:
            self.metadata = raw["metadata"]
        elif "labels" in raw:
            self.metadata = [
                {"label": lbl, "source": "fashionpedia"} for lbl in raw["labels"]
            ]
        else:
            self.metadata = [{} for _ in range(self.embeddings.shape[0])]

    def _normalise_embeddings(self) -> None:
        """L2-normalises the index in place."""
        norms = self.embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.embeddings = self.embeddings / norms

    # ---- search -------------------------------------------------------

    @torch.no_grad()
    def search_looks(self, text_query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Finds the most relevant looks for a text description.

        Runs the CLIP text→image path and (if available) a metadata text path,
        then fuses the two rankings with RRF.

        Args:
            text_query: Free-text fashion description.
            top_k: Number of results to return.

        Returns:
            Result dicts sorted by fused rank, each with ``rank``,
            ``clip_score``, and all metadata keys.
        """
        txt_emb = self.model.encode_text([text_query]).to(self.device).float()
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sims = torch.matmul(txt_emb, self.embeddings.T).squeeze(0)

        # Candidate pool: take a wider slice, then fuse + trim to top_k.
        pool = min(max(top_k * 4, top_k), sims.shape[0])
        clip_scores, clip_idx = torch.topk(sims, k=pool)
        clip_ranking: list[int] = clip_idx.cpu().tolist()
        clip_score_by_idx = dict(zip(clip_ranking, clip_scores.cpu().tolist()))

        ranked_lists: list[list[int]] = [clip_ranking]

        text_ranking = self._text_path_ranking(text_query, clip_ranking)
        if text_ranking is not None:
            ranked_lists.append(text_ranking)

        fused = reciprocal_rank_fusion(ranked_lists, top_k=top_k)

        results: list[dict[str, Any]] = []
        for rank, (idx, fused_score) in enumerate(fused, start=1):
            results.append(
                {
                    "rank": rank,
                    "index": idx,
                    "clip_score": round(clip_score_by_idx.get(idx, 0.0), 4),
                    "fused_score": round(fused_score, 4),
                    **self.metadata[idx],
                }
            )
        logger.info("search_looks(%r) → %d results", text_query, len(results))
        return results

    def _text_path_ranking(
        self, text_query: str, candidate_idx: list[int]
    ) -> list[int] | None:
        """Ranks candidates by sentence-transformer descriptor similarity.

        Args:
            text_query: The original query.
            candidate_idx: Indices to score (the CLIP candidate pool).

        Returns:
            Candidate indices ordered by text similarity, or ``None`` if the
            text model is unavailable or no descriptors exist.
        """
        if self.text_model is None:
            return None
        descriptors = [_build_descriptor(self.metadata[i]) for i in candidate_idx]
        if not any(descriptors):
            return None
        embs = self.text_model.encode(
            [text_query] + [d or "fashion item" for d in descriptors],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_emb, desc_embs = embs[0], embs[1:]
        scores = desc_embs @ query_emb
        order = sorted(range(len(candidate_idx)), key=lambda i: -scores[i])
        return [candidate_idx[i] for i in order]

    @torch.no_grad()
    def search_by_image(
        self, image: torch.Tensor, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Finds the most similar looks for a reference image.

        Args:
            image: Preprocessed image ``(1, C, H, W)`` or ``(C, H, W)``.
            top_k: Number of results.

        Returns:
            Result dicts sorted by descending similarity.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        img_emb = self.model.encode_image(image.to(self.device)).float()
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = torch.matmul(img_emb, self.embeddings.T).squeeze(0)
        k = min(top_k, sims.shape[0])
        scores, indices = torch.topk(sims, k=k)
        return [
            {"rank": r, "index": idx, "similarity": round(s, 4), **self.metadata[idx]}
            for r, (s, idx) in enumerate(
                zip(scores.cpu().tolist(), indices.cpu().tolist()), start=1
            )
        ]


def main() -> None:
    """CLI entry point for quick testing."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Search fashion looks by text.")
    parser.add_argument("query", type=str, help="Text query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--combined", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    retriever = VisualFashionRetriever(
        combined_path=args.combined, checkpoint_path=args.checkpoint
    )
    results = retriever.search_looks(args.query, top_k=args.top_k)
    print(f"\nQuery: {args.query!r}\nTop {len(results)} results:\n")
    for hit in results:
        print(
            f"  #{hit['rank']}  fused={hit['fused_score']:.4f} "
            f"clip={hit['clip_score']:.4f}  {_describe_hit(hit)}"
        )


if __name__ == "__main__":
    main()
