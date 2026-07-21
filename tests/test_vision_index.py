"""Tests for the numpy-based visual index (no torch needed)."""

import numpy as np
import pytest

from fg.vision.index import VisualIndex


def _index():
    emb = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32
    )
    meta = [{"title": "east"}, {"title": "north"}, {"title": "east2"}, {"title": "west"}]
    return VisualIndex(emb, meta)


def test_search_ranks_by_cosine():
    idx = _index()
    hits = idx.search(np.array([1.0, 0.0]), top_k=2)
    assert hits[0]["title"] == "east"          # exact match first
    assert hits[1]["title"] == "east2"          # near match second
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[0]["rank"] == 1


def test_search_handles_2d_query():
    idx = _index()
    hits = idx.search(np.array([[0.0, 1.0]]), top_k=1)
    assert hits[0]["title"] == "north"


def test_length_and_normalization():
    idx = _index()
    assert len(idx) == 4
    norms = np.linalg.norm(idx.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_mismatched_counts_raise():
    with pytest.raises(ValueError):
        VisualIndex(np.zeros((3, 2), dtype=np.float32), [{"a": 1}])


def test_save_and_load_roundtrip(tmp_path):
    idx = _index()
    p = idx.save(tmp_path / "idx.npz")
    loaded = VisualIndex.load(p)
    assert len(loaded) == 4
    assert loaded.search(np.array([1.0, 0.0]), top_k=1)[0]["title"] == "east"


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VisualIndex.load(tmp_path / "nope.npz")
