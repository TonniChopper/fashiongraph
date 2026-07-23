"""Tests for Track B supervision (KG-concept alignment pairs) — synthetic."""

import numpy as np

from fg.kg.schema import Triple
from fg.kg.store import KnowledgeGraph
from fg.training.alignment_pairs import (
    build_concept_vocab,
    by_collection_split,
    clean_concept,
    load_supervision,
    signal_report,
)
from fg.vision.index import VisualIndex


def _kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(":memory:")
    kg.add_triples([
        # Two designers sharing "minimalist"; each with a unique concept too.
        Triple("Jil Sander", "known_for", "minimalist",
               subject_type="brand", object_type="aesthetic", source="t"),
        Triple("Jil Sander", "uses_material", "wool",
               subject_type="brand", object_type="material", source="t"),
        Triple("The Row", "known_for", "minimalist",
               subject_type="brand", object_type="aesthetic", source="t"),
        Triple("The Row", "has_silhouette", "draped",
               subject_type="brand", object_type="silhouette", source="t"),
        # A biographical fact that must NOT become a concept.
        Triple("Jil Sander", "based_in", "Milan",
               subject_type="brand", object_type="city", source="t"),
    ])
    return kg


def _index() -> VisualIndex:
    emb = np.eye(4, dtype=np.float32)
    meta = [
        {"designer": "Jil Sander", "show": "Fall 2025"},
        {"designer": "Jil Sander", "show": "Spring 2025"},
        {"designer": "The Row", "show": "Fall 2025"},
        {"designer": "The Row", "show": "Spring 2025"},
    ]
    return VisualIndex(emb, meta)


def test_clean_concept_filters_noise():
    assert clean_concept("Minimalist") == "minimalist"
    assert clean_concept("wool_blend") == "wool blend"
    assert clean_concept("2026") is None                      # bare year
    assert clean_concept("British Designer of the Year award") is None  # too long
    assert clean_concept("not specified in text") is None


def test_load_supervision_attaches_concepts_not_bio():
    recs = load_supervision(_index(), _kg())
    assert len(recs) == 4
    js = next(r for r in recs if r.designer == "jil sander")
    assert "minimalist" in js.concepts and "wool" in js.concepts
    assert "milan" not in js.concepts                          # based_in excluded


def test_shared_vocab_is_cross_designer_only():
    recs = load_supervision(_index(), _kg())
    vocab = build_concept_vocab(recs, min_designers=2)
    assert vocab == ["minimalist"]                            # only shared concept


def test_by_collection_split_holds_out_whole_shows():
    recs = load_supervision(_index(), _kg())
    train, test = by_collection_split(recs, holdout_frac=0.5, seed=1)
    train_groups = {recs[r].group for r in train}
    test_groups = {recs[r].group for r in test}
    assert train_groups.isdisjoint(test_groups)               # no leakage


def test_signal_report_detects_overlap():
    rep = signal_report(load_supervision(_index(), _kg()))
    assert rep["n_designers"] == 2
    assert rep["shared_concepts"] == 1
    assert rep["mean_pairwise_jaccard"] > 0                    # concepts overlap
    assert rep["images_missing_concepts"] == 0
