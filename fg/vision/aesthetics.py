"""Aesthetic scorer — a learned "does this look good" signal.

A small MLP head over fashion image embeddings, trained on **human pairwise
preference judgments** (Surrey aesthetics dataset: "which of these two looks
better"). This gives the Personal Stylist an actual taste gradient to reason
from, instead of the LLM guessing.

Runtime inference is pure ``numpy`` (no torch), so it loads cheaply inside the
stylist and is fully unit-testable. Training lives in
``fg.training.train_aesthetic`` and exports the weights this class consumes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from fg.config import settings

logger: logging.Logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _default_scorer_path() -> Path:
    """Default location of the trained aesthetic head."""
    return settings.embeddings_dir / "aesthetic_head.npz"


class AestheticScorer:
    """Two-layer MLP that maps an image embedding to an aesthetic score in [0, 1].

    Attributes:
        w1, b1, w2, b2: Numpy weight arrays (first layer + output layer).
        input_dim: Expected embedding dimensionality.
    """

    def __init__(
        self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
    ) -> None:
        """Initializes the scorer from weight arrays.

        Args:
            w1: First-layer weights, shape ``(D, H)``.
            b1: First-layer bias, shape ``(H,)``.
            w2: Output weights, shape ``(H,)`` or ``(H, 1)``.
            b2: Output bias, scalar or shape ``(1,)``.
        """
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32).reshape(-1)
        self.w2 = np.asarray(w2, dtype=np.float32).reshape(-1)
        self.b2 = float(np.asarray(b2, dtype=np.float32).reshape(-1)[0])
        self.input_dim = self.w1.shape[0]

    def score(self, embedding: np.ndarray) -> float:
        """Returns the aesthetic score for one embedding, in ``[0, 1]``.

        Args:
            embedding: An image embedding, shape ``(D,)`` or ``(1, D)``.

        Returns:
            Aesthetic score (higher = better styled).
        """
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        hidden = np.maximum(emb @ self.w1 + self.b1, 0.0)  # ReLU
        raw = float(hidden @ self.w2 + self.b2)
        return float(_sigmoid(raw))

    def score_100(self, embedding: np.ndarray) -> int:
        """Returns the score rescaled to an integer 0–100."""
        return int(round(self.score(embedding) * 100))

    def save(self, path: str | Path | None = None) -> Path:
        """Saves weights to a ``.npz`` file."""
        p = Path(path) if path else _default_scorer_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, w1=self.w1, b1=self.b1, w2=self.w2, b2=np.array([self.b2]))
        return p

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AestheticScorer":
        """Loads a trained scorer from a ``.npz`` file.

        Raises:
            FileNotFoundError: If the file is missing.
        """
        p = Path(path) if path else _default_scorer_path()
        if not p.exists():
            raise FileNotFoundError(
                f"Aesthetic head not found: {p}. Train it with "
                f"`python -m fg.training.train_aesthetic`."
            )
        d = np.load(p, allow_pickle=False)
        return cls(d["w1"], d["b1"], d["w2"], d["b2"])
