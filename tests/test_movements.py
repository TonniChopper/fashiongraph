"""Tests for the cross-domain aesthetic-movement matcher + its look-review wiring."""

import numpy as np

from fg.brain.context_builder import ContextBuilder
from fg.capabilities.personal_stylist.look_review import LookReview
from fg.llm.base import LLM, Message
from fg.vision.aesthetic_movements import MOVEMENTS, MovementMatcher


class _FakeEmbedder:
    """Returns deterministic orthogonal-ish vectors so matching is predictable."""

    def __init__(self, dim=8):
        self.dim = dim
        self._rng = np.random.default_rng(1)

    def encode_texts(self, texts):
        # Stable pseudo-embeddings keyed by text hash.
        out = []
        for t in texts:
            r = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(r.normal(size=self.dim))
        return np.array(out, dtype=np.float32)

    def encode_images(self, images, batch_size=32):
        return np.ones((len(images), self.dim), dtype=np.float32)


def test_matcher_returns_ranked_movements():
    m = MovementMatcher(_FakeEmbedder())
    hits = m.match(np.ones(8), top_k=3)
    assert len(hits) == 3
    names = [n for n, _ in hits]
    assert all(n in MOVEMENTS for n in names)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_matcher_top_k_bounded():
    m = MovementMatcher(_FakeEmbedder())
    assert len(m.match(np.ones(8), top_k=999)) == len(MOVEMENTS)


class _CapturingLLM(LLM):
    model = "fake"

    def __init__(self):
        self.last_messages = []

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "## Silhouette & proportion\nClean."


def test_look_review_includes_movements_in_prompt():
    emb = _FakeEmbedder()
    reviewer = LookReview(
        _CapturingLLM(),
        embedder=emb,
        movement_matcher=MovementMatcher(emb),
        context_builder=ContextBuilder(None),
    )
    reviewer.review(object(), occasion="")
    _, user = reviewer.llm.last_messages
    assert "Aesthetic lineage" in user.content
