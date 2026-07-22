"""Tests for the KG graph-reasoning layer (paths + multi-hop)."""

from fg.kg.reasoning import GraphReasoner, format_path
from fg.kg.schema import Triple
from fg.kg.store import KnowledgeGraph


def _reasoner():
    kg = KnowledgeGraph(":memory:")
    kg.add_triples([
        Triple("Prada", "based_in", "Milan", "brand", "city"),
        Triple("Miu Miu", "part_of", "Prada", "brand", "brand"),
        Triple("Miuccia Prada", "creative_director", "Prada", "designer", "brand"),
        Triple("Raf Simons", "collaborated_with", "Miuccia Prada", "designer", "designer"),
        Triple("Jil Sander", "based_in", "Milan", "brand", "city"),
        Triple("Raf Simons", "creative_director", "Jil Sander", "designer", "brand"),
    ])
    return GraphReasoner(kg)


def test_subjects_with_one_hop_filter():
    r = _reasoner()
    milan = r.subjects_with("based_in", "Milan")
    assert set(milan) == {"Prada", "Jil Sander"}


def test_objects_of():
    r = _reasoner()
    assert r.objects_of("Prada", "based_in") == ["Milan"]


def test_paths_finds_multi_hop_connection():
    r = _reasoner()
    # Raf Simons → Miuccia Prada → Prada  (2 hops)
    paths = r.paths("Raf Simons", "Prada", max_hops=3)
    assert paths
    rendered = [format_path(p) for p in paths]
    assert any("Raf Simons" in x and "Prada" in x for x in rendered)
    # shortest path is 2 hops
    assert min(len(p) for p in paths) == 2


def test_paths_none_when_unreachable():
    r = _reasoner()
    assert r.paths("Prada", "Nonexistent Brand", max_hops=3) == []


def test_format_path_readable():
    r = _reasoner()
    p = r.paths("Miu Miu", "Milan", max_hops=3)[0]
    s = format_path(p)
    assert s.startswith("Miu Miu")
    assert "Milan" in s
    assert "—" in s


def test_two_hop_expands_neighbourhood():
    r = _reasoner()
    facts = r.two_hop("Miu Miu")
    joined = " | ".join(f"{f['subject']} {f['relation']} {f['object']}" for f in facts)
    # one hop: Miu Miu part_of Prada; two hops should reach Prada's Milan fact
    assert "Milan" in joined
