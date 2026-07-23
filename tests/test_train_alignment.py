"""Tests for Track B alignment training — numpy logic + torch smoke test."""

import numpy as np
import pytest

from fg.training.alignment_pairs import LookRecord
from fg.training.train_alignment import (
    bootstrap_ci,
    build_designer_weights,
    designer_topk,
    split_groups_three,
    train_projection,
)
from fg.vision.index import VisualIndex


def _records(n_per=6):
    # Two designers, three collections each; rows 0..
    recs, row = [], 0
    for d in ("dior", "prada"):
        for coll in range(3):
            for _ in range(n_per):
                recs.append(LookRecord(row=row, designer=d, group=f"{d}|c{coll}",
                                       concepts=["minimalist", f"{d}_only"]))
                row += 1
    return recs


def test_split_is_disjoint_and_complete():
    recs = _records()
    fit, val, test = split_groups_three(recs, test_frac=0.34, val_frac=0.34, seed=1)
    allrows = set(fit) | set(val) | set(test)
    assert allrows == {r.row for r in recs}
    assert set(fit).isdisjoint(val) and set(fit).isdisjoint(test) and set(val).isdisjoint(test)
    # No collection spans two splits.
    grp = {r.row: r.group for r in recs}
    assert {grp[r] for r in fit}.isdisjoint({grp[r] for r in test})


def test_designer_topk_perfect_separation():
    # Two clusters on orthogonal axes → kNN recovers designer perfectly.
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    des = np.array(["a", "a", "b", "b"])
    res = designer_topk(emb, des, ref_rows=np.array([0, 2]),
                        query_rows=np.array([1, 3]), neighbors=1)
    assert res["top1"] == 1.0
    assert res["hit1"].tolist() == [1.0, 1.0]


def test_bootstrap_ci_bounds():
    lo, hi = bootstrap_ci(np.array([1.0, 1.0, 0.0, 1.0, 0.0]), iters=500, seed=0)
    assert 0.0 <= lo <= hi <= 1.0


def test_build_designer_weights_labels_and_concepts():
    recs = _records(n_per=1)
    row_des, Wdd, keys = build_designer_weights(recs, "labels")
    assert keys == ["dior", "prada"]
    assert np.allclose(Wdd, np.eye(2))            # only same-designer attracts

    row_des_c, Wdd_c, _ = build_designer_weights(recs, "concepts")
    # Shared "minimalist" (each also has a unique concept) → Jaccard 1/3.
    assert Wdd_c[0, 1] == pytest.approx(1 / 3)
    assert Wdd_c[0, 0] == 1.0                     # self weight on the diagonal


def test_train_projection_runs_and_shapes():
    pytest.importorskip("torch")
    recs = _records()
    rng = np.random.default_rng(0)
    # Designer-separated embeddings so there is real signal to preserve.
    Z = np.zeros((len(recs), 4), dtype=np.float32)
    for i, r in enumerate(recs):
        base = np.array([1, 0, 0, 0]) if r.designer == "dior" else np.array([0, 1, 0, 0])
        Z[i] = base + 0.05 * rng.standard_normal(4)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    row_des, Wdd, _ = build_designer_weights(recs, "labels")
    fit, val, _ = split_groups_three(recs, test_frac=0.34, val_frac=0.34, seed=2)
    designers = np.array([r.designer for r in recs])
    W = train_projection(Z, row_des, Wdd, fit, val, designers,
                         epochs=20, patience=20, seed=0)
    assert W.shape == (4, 4)
    assert np.isfinite(W).all()
