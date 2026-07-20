"""Tests for Reciprocal Rank Fusion (pure-Python, no heavy deps)."""

import pytest

from fg.rag.fusion import reciprocal_rank_fusion


def test_consensus_item_wins():
    """An item ranked highly in both lists should beat single-list leaders."""
    clip = ["a", "b", "c", "d"]
    text = ["b", "a", "e", "f"]
    fused = reciprocal_rank_fusion([clip, text])
    ids = [item for item, _ in fused]
    # 'a' (1,2) and 'b' (2,1) appear near top of both → they lead.
    assert set(ids[:2]) == {"a", "b"}


def test_top_k_truncation():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], top_k=2)
    assert len(fused) == 2


def test_scores_descending():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["x", "y", "z"]])
    assert [i for i, _ in fused] == ["x", "y", "z"]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []


def test_negative_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_deterministic_tie_break():
    """Equal scores break by first appearance, deterministically."""
    a = reciprocal_rank_fusion([["p", "q"], ["r", "s"]])
    b = reciprocal_rank_fusion([["p", "q"], ["r", "s"]])
    assert a == b
