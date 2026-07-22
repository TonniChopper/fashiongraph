"""Tests for runway visual grounding — RunwayLinker + LookReview wiring."""

import numpy as np

from fg.brain.context_builder import ContextBuilder
from fg.capabilities.personal_stylist.look_review import LookReview
from fg.vision.index import VisualIndex
from fg.vision.runway import RunwayLinker


def _runway_index(tmp_path):
    # Three "runway looks": two Rick Owens (dark axis), one Prada (other axis).
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    meta = [
        {"designer": "Rick Owens", "title": "Rick Owens — Fall 2026"},
        {"designer": "Rick Owens", "title": "Rick Owens — Spring 2026"},
        {"designer": "Prada", "title": "Prada — Fall 2026"},
    ]
    p = VisualIndex(emb, meta).save(tmp_path / "runway.npz")
    return p


def test_runway_linker_aggregates_designers(tmp_path):
    linker = RunwayLinker(_runway_index(tmp_path))
    out = linker.link(np.array([1.0, 0.0]), top_k=3, n_designers=2)
    names = [d for d, _ in out["designers"]]
    assert names[0] == "Rick Owens"          # two dark looks aggregate highest
    assert out["nearest"]                     # nearest looks returned


def test_runway_linker_other_axis(tmp_path):
    linker = RunwayLinker(_runway_index(tmp_path))
    out = linker.link(np.array([0.0, 1.0]), top_k=3)
    assert out["designers"][0][0] == "Prada"


# ---- look review wiring ----

class _CapturingLLM:
    model = "vlm"

    def __init__(self):
        self.last_messages = []

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "## Design lineage\nReads Rick Owens."


class _FakeEmbedder:
    def encode_images(self, images, batch_size=32):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_look_review_injects_runway_designers(tmp_path):
    linker = RunwayLinker(_runway_index(tmp_path))
    reviewer = LookReview(
        _CapturingLLM(),
        embedder=_FakeEmbedder(),
        runway_linker=linker,
        context_builder=ContextBuilder(None),
    )
    result = reviewer.review(object(), occasion="")
    _, user = reviewer.llm.last_messages
    assert "Nearest runway looks" in user.content
    assert "Rick Owens" in user.content
    assert result.data.get("garments") == []
