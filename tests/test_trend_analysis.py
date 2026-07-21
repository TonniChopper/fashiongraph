"""Tests for the Trend Analyzer (fake LLM — no network)."""

from fg.brain.context_builder import ContextBuilder
from fg.brain.output_contract import Depth, Format, OutputContract
from fg.brain.router import FashionRouter, Intent
from fg.capabilities.understand.trend_analysis import TrendAnalyzer
from fg.llm.base import LLM, Message


class _CapturingLLM(LLM):
    model = "fake"

    def __init__(self):
        self.last_messages: list[Message] = []

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "## Definition\nQuiet luxury is understated, material-led dressing."


class _FakeRetriever:
    def retrieve(self, query, n_results=5, filters=None):
        return [
            {"document": "Quiet luxury emphasises quality over logos.",
             "metadata": {"title": "Quiet luxury"}, "distance": 0.15}
        ]


def _analyzer():
    return TrendAnalyzer(_CapturingLLM(), ContextBuilder(_FakeRetriever()))


def test_run_returns_analysis_and_sources():
    result = _analyzer().run("quiet luxury", OutputContract(Depth.DETAILED, Format.REPORT))
    assert "Definition" in result.text
    assert "Quiet luxury" in result.sources
    assert result.data["topic"] == "quiet luxury"


def test_prompt_includes_topic_and_grounding():
    a = _analyzer()
    a.run("gorpcore")
    system, user = a.llm.last_messages
    assert "gorpcore" in user.content
    assert "Quiet luxury emphasises" in user.content  # retrieved grounding present


def test_system_prompt_enforces_grounding_discipline():
    a = _analyzer()
    a.run("gorpcore")
    system, _ = a.llm.last_messages
    assert "(inferred)" in system.content
    assert "Never invent" in system.content


def test_coerce_topic_from_dict():
    assert TrendAnalyzer._coerce_topic({"topic": "Y2K revival"}) == "Y2K revival"


def test_router_dispatches_analyze_to_trend_analyzer():
    router = FashionRouter()
    router.register(_analyzer())
    assert router.classify("show me the trend forecast for denim") == Intent.ANALYZE
    result = router.route("what's the trend with quiet luxury")
    assert "Definition" in result.text
