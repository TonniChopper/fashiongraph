"""Tests for the fashion instruction-data builder (Track A)."""

import json

from fg.kg.schema import Triple
from fg.kg.store import KnowledgeGraph
from fg.training.build_instruction_data import (
    _clean_obj,
    caption_tasks,
    kg_paths,
    kg_qa,
)


def _kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(":memory:")
    kg.add_triples([
        Triple("Celine", "founded_by", "Céline Vipiana",
               subject_type="brand", object_type="designer", source="t"),
        Triple("Celine", "based_in", "1978",            # noisy year → must drop
               subject_type="brand", object_type="city", source="t"),
        Triple("Celine", "uses_material", "leather",
               subject_type="brand", object_type="material", source="t"),
        Triple("Dior", "influenced_by", "Celine",
               subject_type="brand", object_type="brand", source="t"),
    ])
    return kg


def test_clean_obj_drops_temporal():
    assert _clean_obj("Paris") == "Paris"
    assert _clean_obj("1978", drop_temporal=True) is None
    assert _clean_obj("Spring 2026", drop_temporal=True) is None
    assert _clean_obj("not specified in text") is None


def test_kg_qa_curated_and_clean():
    qa = kg_qa(_kg(), ["Celine"])
    qs = {ex["messages"][1]["content"] for ex in qa}
    assert "Who founded Celine?" in qs
    assert "What materials is Celine known for using?" in qs
    # the noisy year-based 'based_in' answer must not appear
    for ex in qa:
        assert "1978" not in ex["messages"][2]["content"]


def test_kg_paths_meaningful_only():
    paths = kg_paths(_kg(), ["Dior", "Celine"], max_pairs=5)
    # Dior -influenced_by-> Celine is a meaningful relation → one connection Q/A.
    assert paths and paths[0]["messages"][1]["content"].startswith("How are")
    assert "influenced" in paths[0]["messages"][2]["content"].lower()


def test_caption_tasks(tmp_path):
    p = tmp_path / "caps.jsonl"
    p.write_text(json.dumps({"designer": "Marni", "show": "Fall 2025",
                             "caption": "A boxy wool coat."}) + "\n", encoding="utf-8")
    tasks = caption_tasks(p)
    assert len(tasks) == 1
    assert "Marni" in tasks[0]["messages"][1]["content"]
    assert tasks[0]["messages"][2]["content"] == "A boxy wool coat."
