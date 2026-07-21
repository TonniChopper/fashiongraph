"""Tests for the pure pair-generation helpers."""

from fg.training.pair_sources import (
    parse_surrey_pair_lines,
    sample_pairs_from_scores,
)


def test_parse_surrey_lines():
    lines = ["27_6.jpg 28_3.jpg 1", "5_1.jpg 6_2.jpg 2", "bad line", "a b 3"]
    out = parse_surrey_pair_lines(lines)
    assert out == [("27_6.jpg", "28_3.jpg", 1), ("5_1.jpg", "6_2.jpg", 2)]


def test_sample_pairs_winner_has_higher_score():
    scored = [("a", 8.0), ("b", 3.0), ("c", 5.0)]
    pairs = sample_pairs_from_scores(scored, margin=1.0, max_pairs=50, seed=0)
    score = dict(scored)
    assert pairs  # some pairs generated
    for win, lose in pairs:
        assert score[win] > score[lose]


def test_sample_pairs_respects_margin():
    # All scores within 0.5 → margin 1.0 yields nothing.
    scored = [("a", 5.0), ("b", 5.2), ("c", 5.4)]
    assert sample_pairs_from_scores(scored, margin=1.0, max_pairs=50) == []


def test_sample_pairs_capped():
    scored = [(str(i), float(i)) for i in range(50)]
    pairs = sample_pairs_from_scores(scored, margin=1.0, max_pairs=10)
    assert len(pairs) == 10


def test_sample_pairs_too_few_items():
    assert sample_pairs_from_scores([("a", 1.0)]) == []


def test_sample_pairs_deterministic():
    scored = [(str(i), float(i % 7)) for i in range(30)]
    a = sample_pairs_from_scores(scored, seed=1, max_pairs=20)
    b = sample_pairs_from_scores(scored, seed=1, max_pairs=20)
    assert a == b
