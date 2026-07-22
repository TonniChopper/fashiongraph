"""Tests for Path A — the look→KG entity linker (fakes; no heavy deps)."""

import numpy as np

from fg.brain.context_builder import ContextBuilder
from fg.capabilities.personal_stylist.look_review import LookReview
from fg.kg.schema import Triple
from fg.kg.store import KnowledgeGraph
from fg.llm.base import LLM, Message
from fg.vision.kg_linker import KGEntityLinker


def _kg():
    kg = KnowledgeGraph(":memory:")
    kg.add_triples([
        Triple("Jil Sander", "known_for", "minimalism", "brand", "aesthetic"),
        Triple("Jil Sander", "has_silhouette", "tailored", "brand", "silhouette"),
        Triple("Jil Sander", "based_in", "Hamburg", "brand", "city"),
        Triple("Versace", "known_for", "maximalism", "brand", "aesthetic"),
        Triple("Versace", "uses_material", "gold print", "brand", "material"),
        Triple("Versace", "based_in", "Milan", "brand", "city"),
    ])
    return kg


class _FakeEmbedder:
    """Deterministic embeddings: descriptor mentioning 'minimalism' → axis 0."""

    def encode_texts(self, texts):
        out = []
        for t in texts:
            v = [1.0, 0.0] if "minimalism" in t.lower() else [0.0, 1.0]
            out.append(v)
        return np.array(out, dtype=np.float32)

    def encode_images(self, images, batch_size=32):
        return np.array([[1.0, 0.0]], dtype=np.float32)  # a 'minimalist' look


def test_descriptor_built_from_facts():
    linker = KGEntityLinker(_FakeEmbedder(), _kg(), min_facts=2)
    d = linker._descriptor("Jil Sander")
    assert d.startswith("Jil Sander —")
    assert "minimalism" in d and "tailored" in d


def test_match_ranks_by_shared_space():
    linker = KGEntityLinker(_FakeEmbedder(), _kg(), min_facts=2)
    hits = linker.match(np.array([1.0, 0.0]), top_k=2)  # minimalist look
    assert hits[0][0] == "Jil Sander"
    assert hits[0][1] >= hits[1][1]


def test_link_attaches_lineage_facts():
    linker = KGEntityLinker(_FakeEmbedder(), _kg(), min_facts=2)
    linked = linker.link(np.array([1.0, 0.0]), top_k=1)
    assert linked[0]["entity"] == "Jil Sander"
    assert any("minimalism" in f for f in linked[0]["facts"])


def test_empty_kg_yields_no_matches():
    linker = KGEntityLinker(_FakeEmbedder(), KnowledgeGraph(":memory:"), min_facts=2)
    assert linker.match(np.array([1.0, 0.0])) == []


# ---- integration with look review ----

class _CapturingLLM(LLM):
    model = "fake"

    def __init__(self):
        self.last_messages: list[Message] = []

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "## Design lineage\nReads minimalist."


def test_look_review_injects_associations():
    emb = _FakeEmbedder()
    reviewer = LookReview(
        _CapturingLLM(),
        embedder=emb,
        kg_linker=KGEntityLinker(emb, _kg(), min_facts=2),
        context_builder=ContextBuilder(None),
    )
    result = reviewer.review(object(), occasion="")
    _, user = reviewer.llm.last_messages
    assert "Design-language associations" in user.content
    assert "Jil Sander" in user.content
    assert result.data  # ran end to end
