"""Tests for one-shot ICL link prediction (LLM mocked)."""

from fg.kg.link_prediction import (
    PREDICTED_SOURCE,
    build_prediction_messages,
    predict_links,
)
from fg.kg.schema import Triple
from fg.kg.store import KnowledgeGraph
from fg.llm.base import LLM


def _kg():
    kg = KnowledgeGraph(":memory:")
    kg.add_triples([
        Triple("Jil Sander", "based_in", "Hamburg", "brand", "city"),
        Triple("Jil Sander", "known_for", "minimalism", "brand", "aesthetic"),
    ])
    return kg


class _FakeLLM(LLM):
    model = "fake"

    def __init__(self, reply):
        self._reply = reply

    def chat(self, messages, *, temperature=None, max_tokens=None):
        return self._reply


def test_prompt_has_oneshot_and_known_facts():
    msgs = build_prediction_messages("Jil Sander", ["Jil Sander based in Hamburg"], k=3)
    system, user = msgs
    assert "Example" in user.content            # the one-shot
    assert "Helmut Lang" in user.content         # worked example
    assert "Jil Sander based in Hamburg" in user.content
    assert "MISSING" in system.content


def test_predict_parses_and_tags_source():
    reply = (
        '[{"subject":"Jil Sander","relation":"from_era","object":"1990s",'
        '"object_type":"era"}]'
    )
    preds = predict_links("Jil Sander", _kg(), _FakeLLM(reply), k=5)
    assert len(preds) == 1
    assert preds[0].relation == "from_era"
    assert preds[0].source == PREDICTED_SOURCE


def test_predict_drops_already_known():
    # Model re-proposes a known fact + a new one; only the new survives.
    reply = (
        '[{"subject":"Jil Sander","relation":"known_for","object":"minimalism"},'
        '{"subject":"Jil Sander","relation":"associated_with","object":"quiet luxury"}]'
    )
    preds = predict_links("Jil Sander", _kg(), _FakeLLM(reply), k=5)
    rels = {(p.relation, p.object_key) for p in preds}
    assert ("known_for", "minimalism") not in rels     # already known → dropped
    assert ("associated_with", "quiet luxury") in rels


def test_predict_keeps_only_entity_subject():
    reply = (
        '[{"subject":"Some Other Brand","relation":"based_in","object":"Paris"},'
        '{"subject":"Jil Sander","relation":"based_in","object":"Milan"}]'
    )
    preds = predict_links("Jil Sander", _kg(), _FakeLLM(reply), k=5)
    assert all(p.subject_key == "jil sander" for p in preds)


def test_predict_handles_garbage():
    assert predict_links("Jil Sander", _kg(), _FakeLLM("no json"), k=5) == []
