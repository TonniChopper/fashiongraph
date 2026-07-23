"""Tests for the PerceptionStack composition root."""

from fg.vision.perception import PerceptionStack, _try, build_perception_stack


def test_try_success_and_failure():
    notes = []
    assert _try(notes.append, "x", lambda: 42) == 42
    assert notes == []
    assert _try(notes.append, "boom", lambda: (_ for _ in ()).throw(ValueError("no"))) is None
    assert notes and "boom" in notes[0]


def test_stack_is_dataclass_with_all_slots():
    s = PerceptionStack()
    for attr in ("embedder", "segmenter", "visual_index", "aesthetic_scorer",
                 "movement_matcher", "kg", "kg_linker", "runway_linker"):
        assert hasattr(s, attr)


def test_build_passthrough_embedder_and_degrades():
    # A dummy embedder is passed through; components that need a *real* embedder
    # (movement matcher etc.) fail closed to None with notes — no crash.
    notes = []
    sentinel = object()
    stack = build_perception_stack(embedder=sentinel, on_note=notes.append)
    assert isinstance(stack, PerceptionStack)
    assert stack.embedder is sentinel
    # segmenter/product-index/scorer need heavy libs or files → None here.
    assert stack.movement_matcher is None  # sentinel has no encode_texts


def test_build_without_embedder_does_not_crash():
    stack = build_perception_stack(embedder=None, on_note=lambda m: None)
    assert isinstance(stack, PerceptionStack)
    # embedder load fails in a test env → embedder-dependent parts are None.
    assert stack.movement_matcher is None
    assert stack.kg_linker is None
    assert stack.runway_linker is None
