"""Tests for the fabric knowledge layer."""

from fg.kg.fabrics import FABRICS, FABRIC_SOURCE, add_fabrics_to_kg, fabrics_to_triples
from fg.kg.store import KnowledgeGraph


def test_ontology_nonempty_and_wellformed():
    assert len(FABRICS) >= 20
    for name, f in FABRICS.items():
        assert f["weight"] and f["drape"] and f["warmth"]
        assert f["season"] and f["texture"] and f["properties"]


def test_fabrics_to_triples_relations():
    triples = fabrics_to_triples()
    assert triples
    rels = {t.relation for t in triples}
    assert {"has_property", "has_texture", "suits_season"} <= rels
    assert all(t.source == FABRIC_SOURCE for t in triples)
    # silk should be lightweight, breathable, summer, smooth
    silk = {(t.relation, t.object) for t in triples if t.subject_key == "silk"}
    assert ("suits_season", "summer") in silk
    assert ("has_texture", "smooth") in silk
    assert any(r == "has_property" and "breathable" in o for r, o in silk)


def test_shared_property_nodes_connect_fabrics():
    kg = KnowledgeGraph(":memory:")
    add_fabrics_to_kg(kg)
    # "warm" is a shared property → multiple fabrics point to it.
    warm_fabrics = kg.subjects_with("has_property", "warm")
    assert len(warm_fabrics) >= 2  # e.g. flannel, fleece


def test_add_fabrics_idempotent():
    kg = KnowledgeGraph(":memory:")
    first = add_fabrics_to_kg(kg)
    second = add_fabrics_to_kg(kg)
    assert first > 0 and second == 0
