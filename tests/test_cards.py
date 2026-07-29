"""Tests for typed cards + agent card collection (stub LLM, no network/models)."""

import json

from fg.brain import cards
from fg.brain.agent import ReActAgent


class _JSONLLM:
    """Returns a fixed JSON blob (simulates a capability producing a card)."""

    def __init__(self, payload):
        self.payload = payload

    def chat(self, messages, **kw):
        return "Sure, here it is:\n" + json.dumps(self.payload)

    def complete(self, *a, **k):
        return ""


class _FakeKG:
    def outgoing(self, entity):
        return [
            {"relation": "known_for", "object": "minimalism"},
            {"relation": "uses_material", "object": "wool"},
        ]

    def facts_as_text(self, entity, limit=20):
        return ["Jil Sander known for minimalism", "Jil Sander uses material wool"]


def test_extract_json_tolerates_prose_and_trailing_commas():
    assert cards._extract_json('blah {"a": 1, "b": [2,3,],} tail')["a"] == 1
    assert cards._extract_json("no json here") is None


def test_build_style_card_parses_json():
    payload = {"title": "Autumn minimal", "outfits": [
        {"name": "Look 1", "pieces": [{"slot": "top", "item": "charcoal knit"}],
         "palette": [{"name": "charcoal", "hex": "#36454F"}], "why": "clean"}], "tip": "keep it tonal"}
    card = cards.build_style_card(_JSONLLM(payload), "style me for autumn")
    assert card["type"] == "style"
    assert card["outfits"][0]["palette"][0]["hex"] == "#36454F"


def test_build_style_card_falls_back_to_text():
    class Bad:
        def chat(self, m, **k):
            return "I couldn't make JSON, sorry."
        def complete(self, *a, **k):
            return ""
    card = cards.build_style_card(Bad(), "x")
    assert card["type"] == "style" and card.get("raw") is True and card["text"]


def test_lineage_card_is_deterministic_from_kg():
    card = cards.build_lineage_card(_FakeKG(), "Jil Sander")
    assert card["type"] == "lineage" and card["entity"] == "Jil Sander"
    assert card["by_relation"]["known for"] == ["minimalism"]
    assert cards.build_lineage_card(None, "x") is None


def test_agent_collects_typed_card_from_trend_tool():
    llm = _CardThenFinal({"topic": "quiet luxury", "verdict": "cooling but durable",
                          "score": 62, "trajectory": "plateauing",
                          "evidence_for": ["still on runways"], "evidence_against": ["meme fatigue"]})
    res = ReActAgent(llm, kg=_FakeKG(), max_steps=3).run("rate quiet luxury")
    assert any(c["type"] == "trend" and c["score"] == 62 for c in res.cards)


class _CardThenFinal:
    """Step 1: call trend tool. Step 2 (the tool's own LLM call): return JSON.
    Step 3: Final."""

    def __init__(self, trend_payload):
        self.trend_payload = trend_payload
        self.n = 0

    def chat(self, messages, **kw):
        self.n += 1
        if self.n == 1:
            return "Thought: rate it.\nAction: trend[quiet luxury]"
        if self.n == 2:
            return json.dumps(self.trend_payload)      # the trend card builder's call
        return "Final: Quiet luxury is cooling but durable (62/100)."

    def complete(self, *a, **k):
        return ""
