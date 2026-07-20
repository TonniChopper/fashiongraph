"""Shared text embedding function for the RAG store.

By default ChromaDB uses a bundled ONNX all-MiniLM that runs on **CPU only**.
On Apple Silicon that leaves the GPU idle. This module supplies a
``sentence-transformers`` embedding function pinned to the best available
device (MPS on the M4, CUDA on Colab, else CPU), so the indexer and retriever
both use the GPU — and, crucially, use the *same* function so vectors stay
compatible.

If chromadb / sentence-transformers aren't installed, ``get_text_embedding_function``
returns ``None`` and callers fall back to Chroma's default embedder.
"""

from __future__ import annotations

import logging

from fg.config import settings

logger: logging.Logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    """Resolves the embedding device.

    Args:
        preference: ``"auto"`` (mps→cuda→cpu) or an explicit device string.

    Returns:
        A torch device string. Falls back to ``"cpu"`` if torch is missing.
    """
    if preference and preference != "auto":
        return preference
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — torch absent or probe failed
        pass
    return "cpu"


def get_text_embedding_function(model_name: str | None = None, device: str | None = None):
    """Builds a Chroma-compatible sentence-transformers embedding function.

    Args:
        model_name: HF model id; defaults to ``settings.text_embed_model``.
        device: Device string; defaults to resolving ``settings.embed_device``.

    Returns:
        A Chroma ``SentenceTransformerEmbeddingFunction`` on the chosen device,
        or ``None`` if the required libraries are unavailable (caller then uses
        Chroma's default CPU embedder).
    """
    try:
        from chromadb.utils import embedding_functions
    except ImportError:
        logger.warning("chromadb not installed — using no custom embedder.")
        return None

    dev = device or resolve_device(settings.embed_device)
    try:
        fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name or settings.text_embed_model,
            device=dev,
        )
    except Exception as exc:  # noqa: BLE001 — sentence-transformers missing, etc.
        logger.warning("Falling back to default embedder (%s).", exc)
        return None

    logger.info("Text embedder: %s on device=%s", settings.text_embed_model, dev)
    return fn
