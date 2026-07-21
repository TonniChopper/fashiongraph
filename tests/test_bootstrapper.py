"""Tests for the Brand Bootstrapper (with a fake LLM — no network)."""

from fg.brain.context_builder import ContextBuilder
from fg.brain.output_contract import Depth, Format, OutputContract
from fg.capabilities.strategize.bootstrapper import QUESTIONS, BrandBootstrapper
from fg.llm.base import LLM, Message


class _CapturingLLM(LLM):
    model = "fake"

    def __init__(self):
        self.last_messages: list[Message] = []

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "## Brand DNA\nAria — quiet luxury knitwear."


class _FakeRetriever:
    def retrieve(self, query, n_results=5, filters=None):
        return [
            {"document": "Quiet luxury favours material over logos.",
             "metadata": {"title": "Quiet luxury"}, "distance": 0.2}
        ]


def _bootstrapper():
    return BrandBootstrapper(_CapturingLLM(), ContextBuilder(_FakeRetriever()))


def test_has_ten_questions():
    assert len(QUESTIONS) == 10
    assert QUESTIONS[0].id == "working_name"


def test_run_returns_document_and_sources():
    bs = _bootstrapper()
    result = bs.run(
        {"aesthetic": "quiet luxury", "category": "knitwear"},
        OutputContract(Depth.DETAILED, Format.REPORT),
    )
    assert "Brand DNA" in result.text
    assert "Quiet luxury" in result.sources


def test_run_grounds_prompt_with_answers_and_context():
    bs = _bootstrapper()
    bs.run({"aesthetic": "gorpcore", "inspirations": "Arc'teryx"})
    system, user = bs.llm.last_messages
    assert system.role == "system"
    assert "gorpcore" in user.content
    assert "Quiet luxury favours" in user.content  # retrieved grounding present
    assert "(inferred)" in system.content  # grounding discipline applied


def test_retrieval_query_prefers_semantic_fields():
    q = BrandBootstrapper._retrieval_query(
        {"aesthetic": "romantic minimalism", "price_tier": "luxury"}
    )
    assert "romantic minimalism" in q
    assert "luxury" not in q  # price_tier is not a semantic retrieval field


def test_free_text_request_is_coerced():
    assert BrandBootstrapper._coerce_answers("archival streetwear")["aesthetic"] == (
        "archival streetwear"
    )


def test_run_from_free_text():
    bs = _bootstrapper()
    result = bs.run("minimalist tailoring label")
    assert result.text  # still produces a document
