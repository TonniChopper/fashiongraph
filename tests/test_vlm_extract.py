"""Tests for VLM visual extraction (pure parse + triple conversion)."""

from fg.vision.vlm_extract import (
    VLM_SOURCE,
    build_extraction_prompt,
    look_to_triples,
    parse_look,
)


def test_prompt_attaches_image_and_labels():
    msgs = build_extraction_prompt("Rick Owens", "Fall 2026", "IMGB64")
    system, user = msgs
    assert "JSON" in system.content
    assert "Rick Owens" in user.content and "Fall 2026" in user.content
    assert user.images == ["IMGB64"]


def test_parse_look_valid():
    raw = (
        'Sure: {"caption":"A dark draped column.","silhouettes":["column","draped"],'
        '"materials":["leather","jersey"],"aesthetics":["gothic","minimalist"],'
        '"garments":["gown"],"palette":["black"]}'
    )
    look = parse_look(raw)
    assert look["caption"].startswith("A dark")
    assert look["silhouettes"] == ["column", "draped"]
    assert "leather" in look["materials"]


def test_parse_look_handles_garbage():
    assert parse_look("no json")["silhouettes"] == []
    assert parse_look("")["caption"] == ""


def test_parse_look_coerces_string_to_list():
    look = parse_look('{"materials":"wool","aesthetics":["utilitarian"]}')
    assert look["materials"] == ["wool"]


def test_look_to_triples():
    look = {"silhouettes": ["column"], "materials": ["leather"],
            "aesthetics": ["gothic"], "garments": [], "palette": ["black"], "caption": ""}
    triples = look_to_triples(look, "Rick Owens", season="Fall 2026")
    rels = {(t.relation, t.object) for t in triples}
    assert ("has_silhouette", "column") in rels
    assert ("uses_material", "leather") in rels
    assert ("known_for", "gothic") in rels
    assert ("from_era", "Fall 2026") in rels
    assert all(t.source == VLM_SOURCE for t in triples)


def test_look_to_triples_empty():
    assert look_to_triples({"silhouettes": [], "materials": [], "aesthetics": []}, "X") == []
