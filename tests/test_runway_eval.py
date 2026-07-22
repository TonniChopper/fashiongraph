"""Tests for runway grounding eval (designer top-k accuracy)."""

import numpy as np
import pytest

from fg.vision.index import VisualIndex
from fg.vision.runway_eval import evaluate_designer_topk


def _clustered_index(per_designer=20, dim=8, seed=0):
    """Build an index where each designer is a tight cluster (separable)."""
    rng = np.random.default_rng(seed)
    designers = ["A", "B", "C"]
    embs, meta = [], []
    for d, center in zip(designers, np.eye(3, dim)):
        for _ in range(per_designer):
            v = center + rng.normal(scale=0.05, size=dim)
            embs.append(v)
            meta.append({"designer": d})
    return VisualIndex(np.array(embs, dtype=np.float32), meta)


def test_separable_designers_score_high():
    res = evaluate_designer_topk(_clustered_index(), holdout_frac=0.3, neighbors=5)
    assert res["top1"] > 0.9                 # tight clusters → easy
    assert res["n_designers"] == 3
    assert res["random_top1"] == round(1 / 3, 3)


def test_topk_monotonic():
    res = evaluate_designer_topk(_clustered_index(), holdout_frac=0.3)
    assert res["top1"] <= res["top3"] <= res["top5"]


def test_too_small_raises():
    idx = VisualIndex(np.eye(3, 4, dtype=np.float32), [{"designer": x} for x in "ABC"])
    with pytest.raises(ValueError):
        evaluate_designer_topk(idx)
