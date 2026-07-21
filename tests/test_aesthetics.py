"""Tests for the numpy aesthetic scorer + its wiring into look review."""

import numpy as np

from fg.brain.context_builder import ContextBuilder
from fg.capabilities.personal_stylist.look_review import LookReview
from fg.llm.base import LLM, Message
from fg.vision.aesthetics import AestheticScorer


def _scorer(dim=4, hidden=3):
    rng = np.random.default_rng(0)
    return AestheticScorer(
        w1=rng.normal(size=(dim, hidden)),
        b1=np.zeros(hidden),
        w2=rng.normal(size=(hidden,)),
        b2=np.array([0.0]),
    )


def test_score_in_unit_interval():
    s = _scorer()
    val = s.score(np.ones(4))
    assert 0.0 <= val <= 1.0


def test_score_100_is_int_0_100():
    s = _scorer()
    v = s.score_100(np.ones(4))
    assert isinstance(v, int) and 0 <= v <= 100


def test_score_is_deterministic():
    s = _scorer()
    emb = np.array([0.1, -0.2, 0.3, 0.4])
    assert s.score(emb) == s.score(emb)


def test_higher_weights_shift_score(monkeypatch):
    # A strongly positive linear path should score high; negative should score low.
    hi = AestheticScorer(w1=np.eye(2), b1=np.zeros(2),
                         w2=np.array([10.0, 10.0]), b2=np.array([0.0]))
    lo = AestheticScorer(w1=np.eye(2), b1=np.zeros(2),
                         w2=np.array([-10.0, -10.0]), b2=np.array([0.0]))
    emb = np.array([1.0, 1.0])
    assert hi.score(emb) > 0.9
    assert lo.score(emb) < 0.5  # ReLU zeros the negative path → sigmoid(0)=0.5


def test_save_load_roundtrip(tmp_path):
    s = _scorer()
    p = s.save(tmp_path / "head.npz")
    loaded = AestheticScorer.load(p)
    emb = np.array([0.2, 0.4, -0.1, 0.0])
    assert abs(loaded.score(emb) - s.score(emb)) < 1e-6


# ---- wiring into look review ----

class _CapturingLLM(LLM):
    model = "fake"

    def __init__(self):
        self.last_messages = []

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "## Silhouette & proportion\nStrong line."


class _FakeEmbedder:
    def encode_images(self, images, batch_size=32):
        return np.ones((1, 4), dtype=np.float32)


def test_look_review_uses_aesthetic_score():
    reviewer = LookReview(
        _CapturingLLM(),
        embedder=_FakeEmbedder(),
        aesthetic_scorer=_scorer(),
        context_builder=ContextBuilder(None),
    )
    result = reviewer.review(object(), occasion="")
    # score computed and surfaced into the prompt
    _, user = reviewer.llm.last_messages
    assert "/100" in user.content
    # rubric present in system prompt
    system, _ = reviewer.llm.last_messages
    assert "proportion" in system.content.lower()
