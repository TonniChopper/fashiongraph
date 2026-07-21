"""Tests for the Personal Stylist look-review capability (fakes, no vision libs)."""

import numpy as np

from fg.brain.context_builder import ContextBuilder
from fg.brain.output_contract import Depth, Format, OutputContract
from fg.capabilities.personal_stylist.look_review import LookReview, Perception
from fg.llm.base import LLM, Message


class _CapturingLLM(LLM):
    model = "fake"

    def __init__(self):
        self.last_messages: list[Message] = []

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "## Silhouette & proportion\nBalanced, elongated line."


class _FakeSegmenter:
    def labels(self, image, min_area=0.02):
        return ["Upper-clothes", "Pants", "Left-shoe"]


class _FakeEmbedder:
    def encode_images(self, images, batch_size=32):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class _FakeIndex:
    def search(self, vec, top_k=5):
        return [{"title": "Charcoal wool trousers", "colour": "Grey", "score": 0.82}]


class _FakeRetriever:
    def retrieve(self, query, n_results=5, filters=None):
        return [{"document": "Monochrome palettes elongate the frame.",
                 "metadata": {"title": "Proportion"}, "distance": 0.2}]


def _reviewer(**kw):
    return LookReview(_CapturingLLM(), context_builder=ContextBuilder(_FakeRetriever()), **kw)


def test_perception_summary_helpers():
    p = Perception(garments=["Dress"], similar=[{"title": "Silk slip", "colour": "Ivory", "score": 0.7}])
    assert "Dress" in p.garments_text()
    assert "Silk slip" in p.similar_text()


def test_review_full_perception():
    r = _reviewer(segmenter=_FakeSegmenter(), embedder=_FakeEmbedder(), visual_index=_FakeIndex())
    result = r.review(object(), occasion="job interview", contract=OutputContract(Depth.DETAILED, Format.REPORT))
    assert "Silhouette" in result.text
    assert result.data["garments"] == ["Upper-clothes", "Pants", "Left-shoe"]
    assert "Charcoal wool trousers" in result.sources
    # prompt carried garments + occasion + grounding
    system, user = r.llm.last_messages
    assert "Upper-clothes" in user.content
    assert "job interview" in user.content
    assert "(inferred)" in system.content


def test_review_degrades_without_vision():
    r = _reviewer()  # no segmenter / embedder / index
    result = r.review(object(), occasion="")
    assert result.text  # still produces a review from RAG only
    assert result.data["garments"] == []


def test_intent_is_style():
    assert LookReview.intents == ("style",)


def test_retrieval_query_includes_garments_and_occasion():
    q = LookReview._retrieval_query(Perception(garments=["Dress", "Belt"]), "gala")
    assert "Dress" in q and "gala" in q
