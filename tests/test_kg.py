"""Tests for the knowledge graph: schema, SQLite store, extractor parsing,
and context-builder KG grounding."""

from fg.brain.context_builder import ContextBuilder
from fg.kg.evaluate import fact_coverage, parse_judge_verdict
from fg.kg.extractor import parse_triples
from fg.kg.schema import Triple, canonical_relation, normalize_entity
from fg.kg.store import KnowledgeGraph


# ---- schema ----

def test_normalize_entity():
    assert normalize_entity("  Maison   Margiela ") == "maison margiela"
    assert normalize_entity("PRADA") == "prada"


def test_normalize_entity_folds_accents_and_punct():
    # The two failure modes we diagnosed: accents and underscores.
    assert normalize_entity("Céline") == normalize_entity("Celine") == "celine"
    assert normalize_entity("Christian_Dior") == normalize_entity("Christian Dior")
    assert normalize_entity("Comme des Garçons") == "comme des garcons"
    assert normalize_entity("Yves Saint Laurent") == "yves saint laurent"


def test_triple_validity():
    assert Triple("Prada", "based_in", "Milan").is_valid()
    assert not Triple("Prada", "not_a_relation", "Milan").is_valid()
    assert not Triple("Prada", "based_in", "prada").is_valid()  # self-loop
    assert not Triple("", "based_in", "Milan").is_valid()


# ---- store ----

def _kg():
    kg = KnowledgeGraph(":memory:")
    kg.add_triples([
        Triple("Prada", "based_in", "Milan", "brand", "city", "prada"),
        Triple("Prada", "creative_director", "Miuccia Prada", "brand", "designer", "prada"),
        Triple("Miu Miu", "part_of", "Prada", "brand", "brand", "prada"),
    ])
    return kg


def test_store_add_and_count():
    assert _kg().count() == 3


def test_store_idempotent():
    kg = _kg()
    added = kg.add_triples([Triple("Prada", "based_in", "Milan")])  # dup
    assert added == 0
    assert kg.count() == 3


def test_store_neighbors_by_subject_and_object():
    kg = _kg()
    facts = kg.facts_as_text("Prada")
    joined = " | ".join(facts)
    assert "based in Milan" in joined
    assert "Miu Miu part of Prada" in joined  # matched as object


def test_store_case_insensitive_lookup():
    assert _kg().facts_as_text("prada")  # lower-case query still matches


def test_store_entities_and_stats():
    kg = _kg()
    ents = kg.entities()
    assert "prada" in ents and "milan" in ents
    s = kg.stats()
    assert s["triples"] == 3
    assert s["relations"]["based_in"] == 1


# ---- extractor parsing (pure) ----

def test_parse_triples_valid_json():
    raw = (
        'Here you go: [{"subject":"Prada","subject_type":"brand",'
        '"relation":"based_in","object":"Milan","object_type":"city"}]'
    )
    out = parse_triples(raw, source="wiki")
    assert len(out) == 1
    assert out[0].relation == "based_in"
    assert out[0].source == "wiki"


def test_parse_triples_drops_bad_relation():
    raw = '[{"subject":"A","relation":"loves","object":"B"}]'
    assert parse_triples(raw) == []


def test_canonical_relation_maps_synonyms():
    assert canonical_relation("located_in") == "based_in"
    assert canonical_relation("Inspired By") == "influenced_by"
    assert canonical_relation("designed-by") == "creative_director"
    assert canonical_relation("known_for") == "known_for"
    assert canonical_relation("totally_unknown") == ""


def test_parse_triples_recovers_via_synonym():
    raw = '[{"subject":"Prada","relation":"located_in","object":"Milan"}]'
    out = parse_triples(raw)
    assert len(out) == 1 and out[0].relation == "based_in"


def test_fact_coverage():
    ans = "Prada is based in Milan and known for minimalism."
    assert fact_coverage(ans, ["Milan", "minimalism"]) == 1.0
    assert fact_coverage(ans, ["Milan", "leather"]) == 0.5
    assert fact_coverage(ans, []) == 0.0


def test_parse_judge_verdict():
    assert parse_judge_verdict("A") == "A"
    assert parse_judge_verdict("TIE") == "tie"
    assert parse_judge_verdict("") == "tie"
    # Reasoned replies ending in a VERDICT line (the new format).
    assert parse_judge_verdict("A lists more facts.\nVERDICT: A") == "A"
    assert parse_judge_verdict("B is richer and specific.\nVERDICT: B") == "B"
    assert parse_judge_verdict("Both similar.\nVERDICT: TIE") == "tie"
    # The verdict line wins even if reasoning mentions the other letter.
    assert parse_judge_verdict("Answer A is vague; B wins.\nVERDICT: B") == "B"


def test_parse_triples_handles_garbage():
    assert parse_triples("no json here") == []
    assert parse_triples("") == []


# ---- context builder KG grounding ----

def test_context_builder_injects_kg_facts():
    kg = _kg()
    ctx = ContextBuilder(retriever=None, kg=kg).build("tell me about prada")
    assert ctx.kg_facts  # matched 'prada'
    block = ctx.knowledge_block()
    assert "Knowledge-graph facts" in block
    assert "Milan" in block


def test_context_builder_no_kg_match():
    ctx = ContextBuilder(retriever=None, kg=_kg()).build("something unrelated")
    assert ctx.kg_facts == []
