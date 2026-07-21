"""Fashion image/text embedder — Marqo-FashionSigLIP via OpenCLIP.

Marqo-FashionSigLIP is SOTA on fashion retrieval (+57% recall@1 vs FashionCLIP
2.0) and loads through OpenCLIP's ``hf-hub:`` interface. Produces L2-normalised
embeddings in a shared image/text space, so text queries and image queries hit
the same index.

All public methods return ``numpy`` arrays so downstream code (the visual index,
tests) needs no torch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from fg.config import settings

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

logger: logging.Logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    """Resolves the compute device (mps→cuda→cpu), falling back if torch absent."""
    if preference and preference != "auto":
        return preference
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class FashionEmbedder:
    """Wraps a Marqo fashion CLIP/SigLIP model for image + text embedding.

    Attributes:
        model_name: HF model id (e.g. ``Marqo/marqo-fashionSigLIP``).
        device: Torch device string.
        dim: Embedding dimensionality (known after first encode).
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        """Loads the OpenCLIP model + preprocessing + tokenizer.

        Args:
            model_name: HF model id; defaults to ``settings.fashion_embed_model``.
            device: Device string; auto-resolved if ``None``.

        Raises:
            RuntimeError: If OpenCLIP / the model cannot be loaded.
        """
        self.model_name = model_name or settings.fashion_embed_model
        self.device = device or resolve_device(settings.embed_device)
        try:
            import open_clip
            import torch

            hub = f"hf-hub:{self.model_name}"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(hub)
            self.tokenizer = open_clip.get_tokenizer(hub)
            self.model = self.model.to(self.device).eval()
            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load fashion embedder '{self.model_name}'. "
                f"Install extras: pip install open-clip-torch torch. ({exc})"
            ) from exc
        self.dim: int | None = None
        logger.info("FashionEmbedder '%s' on %s", self.model_name, self.device)

    def encode_images(self, images: "list[Image]", batch_size: int = 32) -> np.ndarray:
        """Embeds a list of PIL images.

        Args:
            images: PIL images.
            batch_size: Batch size for the forward pass.

        Returns:
            L2-normalised embeddings, shape ``(N, D)``.
        """
        torch = self._torch
        vecs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            tensors = torch.stack([self.preprocess(im.convert("RGB")) for im in batch])
            tensors = tensors.to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(tensors)
            vecs.append(self._normalize(feats))
        out = np.concatenate(vecs, axis=0) if vecs else np.empty((0, 0), dtype=np.float32)
        self.dim = out.shape[1] if out.size else self.dim
        return out

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Embeds a list of text strings.

        Args:
            texts: Query / caption strings.

        Returns:
            L2-normalised embeddings, shape ``(N, D)``.
        """
        torch = self._torch
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        out = self._normalize(feats)
        self.dim = out.shape[1]
        return out

    def _normalize(self, feats: Any) -> np.ndarray:
        """L2-normalises a torch tensor and returns float32 numpy."""
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feats.float().cpu().numpy().astype(np.float32)
