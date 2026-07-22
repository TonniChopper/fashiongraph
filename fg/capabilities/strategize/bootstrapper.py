"""Brand Bootstrapper — AI co-founder that builds a brand from ~10 answers.

Pure LLM + RAG (no ML training). Given short answers about the founder's vision,
it retrieves relevant fashion knowledge and produces Brand DNA, positioning /
strategy, and a starter-collection brief. This is the Phase-2 flagship demo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fg.brain.context_builder import ContextBuilder, FusionContext
from fg.brain.output_contract import Depth, Format, OutputContract
from fg.capabilities._prompts import GROUNDING_DISCIPLINE
from fg.capabilities.base import Capability, CapabilityResult
from fg.llm.base import LLM, Message

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Question:
    """One onboarding question.

    Attributes:
        id: Answer key.
        prompt: Question text shown to the founder.
        hint: Optional example / guidance.
    """

    id: str
    prompt: str
    hint: str = ""


#: The 10-question brand-from-scratch flow.
QUESTIONS: tuple[Question, ...] = (
    Question("working_name", "Do you have a working brand name or a word you love?",
             "Totally fine to say 'none yet'."),
    Question("category", "What will you make?",
             "e.g. womenswear ready-to-wear, sneakers, knitwear, accessories"),
    Question("audience", "Who is it for?",
             "age, lifestyle, and what they value"),
    Question("price_tier", "What price tier?",
             "accessible / contemporary / premium / luxury"),
    Question("aesthetic", "Describe the aesthetic in a few words.",
             "e.g. quiet luxury, gorpcore, romantic minimalism, archival streetwear"),
    Question("values", "What does the brand stand for?",
             "e.g. craft, sustainability, inclusivity, irreverence"),
    Question("signature", "Any signature element you want to own?",
             "a silhouette, material, construction detail, or motif"),
    Question("palette", "Preferred colours and materials?",
             "e.g. bone/charcoal/oxblood; heavy wool, dry cotton, deadstock silk"),
    Question("inspirations", "Brands, designers, eras, or cultural references you admire?",
             "e.g. early Helmut Lang, 90s Prada, Yohji, 1970s Paris"),
    Question("differentiator", "What makes it different from what's already out there?", ""),
)


class BrandBootstrapper(Capability):
    """Generates a brand foundation from onboarding answers.

    Attributes:
        name: Capability id.
        intents: Router intents served.
        llm: The language model backend.
        context_builder: Supplies RAG grounding.
    """

    name = "brand_bootstrapper"
    intents = ("bootstrap",)

    def __init__(self, llm: LLM, context_builder: ContextBuilder | None = None) -> None:
        """Initializes the bootstrapper.

        Args:
            llm: LLM backend (Ollama / API).
            context_builder: Optional context builder; ungrounded if ``None``.
        """
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder(None)

    def questions(self) -> tuple[Question, ...]:
        """Returns the onboarding questions."""
        return QUESTIONS

    def run(
        self, request: Any, contract: OutputContract | None = None
    ) -> CapabilityResult:
        """Builds the brand foundation.

        Args:
            request: Either an answers ``dict`` (keys = question ids) or a free-
                text idea string (used as the aesthetic seed).
            contract: Output policy; defaults to detailed report.

        Returns:
            A :class:`CapabilityResult` whose ``text`` is the brand document.
        """
        answers = self._coerce_answers(request)
        contract = contract or OutputContract(Depth.DETAILED, Format.REPORT)

        query = self._retrieval_query(answers)
        ctx = self.context_builder.build(query, n_rag=6)

        messages = self._build_messages(answers, ctx, contract)
        text = self.llm.chat(messages, max_tokens=1400)

        sources = sorted(
            {
                (c.get("metadata", {}).get("title")
                 or c.get("metadata", {}).get("source", "source"))
                for c in ctx.rag_chunks
            }
        )
        return CapabilityResult(text=text, data={"answers": answers}, sources=sources)

    # ---- internals ----------------------------------------------------

    @staticmethod
    def _coerce_answers(request: Any) -> dict[str, str]:
        """Normalizes the request into an answers dict."""
        if isinstance(request, dict):
            return {k: str(v) for k, v in request.items()}
        # Free-text: seed the aesthetic field so the flow still runs.
        return {"aesthetic": str(request)}

    @staticmethod
    def _retrieval_query(answers: dict[str, str]) -> str:
        """Builds a RAG query from the most semantic answer fields."""
        parts = [
            answers.get(k, "")
            for k in ("aesthetic", "inspirations", "category", "values")
        ]
        return " ".join(p for p in parts if p).strip() or "fashion brand identity"

    def _build_messages(
        self,
        answers: dict[str, str],
        ctx: FusionContext,
        contract: OutputContract,
    ) -> list[Message]:
        """Assembles the chat messages for generation."""
        system = (
            "You are FashionGraph's Brand Bootstrapper — an AI co-founder with the "
            "eye of a creative director and the rigor of a brand strategist. You "
            "help independent designers turn a rough vision into a coherent brand. "
            "Be specific and opinionated, never generic. "
            + contract.style_directive() + " " + GROUNDING_DISCIPLINE
        )

        answers_block = "\n".join(
            f"- {q.prompt} → {answers.get(q.id, '(not provided)')}"
            for q in QUESTIONS
        )
        knowledge = ctx.knowledge_block() or "(no external context retrieved)"

        user = (
            f"## Founder's answers\n{answers_block}\n\n"
            f"## Fashion knowledge (for grounding)\n{knowledge}\n\n"
            "## Task\n"
            "Design this brand. Produce these sections:\n"
            "1. **Brand DNA** — one-line essence, positioning statement, 3–5 "
            "aesthetic codes, tone of voice.\n"
            "2. **Positioning & Strategy** — target customer, price tier, where it "
            "sits vs. references, launch channel, seasonal cadence.\n"
            "3. **Signature** — the ownable element and how it recurs.\n"
            "4. **Starter Collection** — 8–12 pieces with a one-line rationale each, "
            "consistent with the DNA. Vary how the signature shows up across pieces "
            "(some overt, some whisper-subtle, a few without it) so the line has "
            "range and rhythm — do not repeat the same detail on every item.\n"
            "5. **Next Steps** — the first 3 concrete moves to launch.\n"
            "If the founder left something blank, make a strong, justified "
            "recommendation rather than asking."
        )
        return [Message("system", system), Message("user", user)]
