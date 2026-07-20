"""Tests for the brain: output contract, context builder, memory, router."""

import pytest

from fg.brain.context_builder import ContextBuilder, FusionContext
from fg.brain.memory import Memory
from fg.brain.output_contract import Depth, Format, OutputContract
from fg.brain.router import FashionRouter, Intent
from fg.capabilities.base import Capability, CapabilityResult
from fg.llm.base import LLM, Message


# ---- output contract ----

def test_contract_directive_varies_by_depth():
    surface = OutputContract(Depth.SURFACE, Format.CHAT).style_directive()
    expert = OutputContract(Depth.EXPERT, Format.REPORT).style_directive()
    assert surface != expert
    assert "concise" in surface.lower()


def test_contract_from_strings_and_bad():
    assert OutputContract.from_strings("expert", "chat").depth == Depth.EXPERT
    with pytest.raises(ValueError):
        OutputContract.from_strings("bogus", "chat")


# ---- context builder ----

class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, n_results=5, filters=None):
        return self._chunks[:n_results]


def test_context_builder_grounded():
    chunks = [{"document": "Wool is warm.", "metadata": {"title": "Wool"}, "distance": 0.1}]
    ctx = ContextBuilder(_FakeRetriever(chunks)).build("wool", n_rag=3)
    assert "[Wool] Wool is warm." in ctx.rag_text()


def test_context_builder_handles_retriever_failure():
    class _Boom:
        def retrieve(self, *a, **k):
            raise RuntimeError("down")

    ctx = ContextBuilder(_Boom()).build("x")
    assert ctx.rag_chunks == []  # degrades gracefully


def test_context_builder_no_retriever():
    ctx = ContextBuilder(None).build("x")
    assert isinstance(ctx, FusionContext)
    assert ctx.rag_text() == ""


# ---- memory ----

def test_memory_remember_recall_update():
    m = Memory(namespace="test", persist=False)
    m.remember("brand", "Aria")
    m.update({"tier": "luxury"})
    assert m.recall("brand") == "Aria"
    assert m.snapshot() == {"brand": "Aria", "tier": "luxury"}
    assert m.recall("missing", "default") == "default"


# ---- router ----

class _FakeLLM(LLM):
    model = "fake"

    def __init__(self, reply="analyze"):
        self._reply = reply

    def chat(self, messages, *, temperature=None, max_tokens=None):
        return self._reply


class _StubCap(Capability):
    name = "stub"
    intents = ("bootstrap",)

    def run(self, request, contract=None):
        return CapabilityResult(text=f"ran:{request}")


def test_router_keyword_classify():
    r = FashionRouter()
    assert r.classify("I want to start a brand from scratch") == Intent.BOOTSTRAP
    assert r.classify("what should i wear tonight") == Intent.STYLE
    assert r.classify("show me the trend forecast") == Intent.ANALYZE


def test_router_unknown_without_llm():
    assert FashionRouter().classify("the weather is nice") == Intent.UNKNOWN


def test_router_llm_fallback():
    r = FashionRouter(llm=_FakeLLM(reply="design"))
    assert r.classify("qwerty zxcv") == Intent.DESIGN


def test_router_route_dispatches():
    r = FashionRouter()
    r.register(_StubCap())
    result = r.route("bootstrap my label")
    assert result.text.startswith("ran:")


def test_router_route_unregistered_raises():
    with pytest.raises(LookupError):
        FashionRouter().route("what should i wear")
