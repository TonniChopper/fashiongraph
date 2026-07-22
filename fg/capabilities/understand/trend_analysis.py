"""Trend Analysis — cultural-forecaster read on a trend, era, or aesthetic.

Pure LLM + RAG. Given a topic ("quiet luxury", "gorpcore", "1970s Paris"),
it retrieves relevant knowledge and produces a structured analysis: what it is,
where it came from, where it's heading, and what a brand should do about it.
"""

from __future__ import annotations

import logging
from typing import Any

from fg.brain.context_builder import ContextBuilder, FusionContext
from fg.brain.output_contract import Depth, Format, OutputContract
from fg.capabilities._prompts import GROUNDING_DISCIPLINE
from fg.capabilities.base import Capability, CapabilityResult
from fg.llm.base import LLM, Message

logger: logging.Logger = logging.getLogger(__name__)


class TrendAnalyzer(Capability):
    """Analyses a fashion trend / aesthetic / era, grounded in the knowledge core.

    Attributes:
        name: Capability id.
        intents: Router intents served.
        llm: The language model backend.
        context_builder: Supplies RAG grounding.
    """

    name = "trend_analyzer"
    intents = ("analyze",)

    def __init__(self, llm: LLM, context_builder: ContextBuilder | None = None) -> None:
        """Initializes the analyzer.

        Args:
            llm: LLM backend.
            context_builder: Optional context builder; ungrounded if ``None``.
        """
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder(None)

    def run(
        self, request: Any, contract: OutputContract | None = None
    ) -> CapabilityResult:
        """Produces a trend analysis for the given topic.

        Args:
            request: The trend/era/aesthetic as a string (dicts are coerced via
                a ``topic`` key or ``str``).
            contract: Output policy; defaults to detailed report.

        Returns:
            A :class:`CapabilityResult` whose ``text`` is the analysis.
        """
        topic = self._coerce_topic(request)
        contract = contract or OutputContract(Depth.DETAILED, Format.REPORT)

        ctx = self.context_builder.build(topic, n_rag=8)
        messages = self._build_messages(topic, ctx, contract)
        text = self.llm.chat(messages, max_tokens=1200)

        sources = sorted(
            {
                (c.get("metadata", {}).get("title")
                 or c.get("metadata", {}).get("source", "source"))
                for c in ctx.rag_chunks
            }
        )
        return CapabilityResult(text=text, data={"topic": topic}, sources=sources)

    @staticmethod
    def _coerce_topic(request: Any) -> str:
        """Normalizes the request into a topic string."""
        if isinstance(request, dict):
            return str(request.get("topic") or request.get("query") or request)
        return str(request)

    def _build_messages(
        self, topic: str, ctx: FusionContext, contract: OutputContract
    ) -> list[Message]:
        """Assembles the chat messages for the analysis."""
        system = (
            "You are FashionGraph's Trend Analyst — a cultural forecaster who reads "
            "fashion the way an editor and a strategist would. Distinguish signal "
            "from hype. " + contract.style_directive() + " " + GROUNDING_DISCIPLINE
        )
        knowledge = ctx.knowledge_block() or "(no external context retrieved)"
        user = (
            f"## Topic\n{topic}\n\n"
            f"## Fashion knowledge (for grounding)\n{knowledge}\n\n"
            "## Task\nAnalyse this trend. Produce these sections:\n"
            "1. **Definition** — what it is and its visual signatures.\n"
            "2. **Origins & cultural context** — where it came from and what drove it.\n"
            "3. **Trajectory** — emerging, peaking, or declining, and the evidence.\n"
            "4. **Drivers** — the cultural/economic forces sustaining it.\n"
            "5. **Key players** — brands, designers, and references that embody it. "
            "Name only brands you are confident actually belong to this trend; if "
            "unsure, mark them '(inferred)' or describe the archetype instead.\n"
            "6. **Implications** — what a brand or designer should do (opportunity, "
            "risk, timing).\n"
            "7. **Adjacent & counter-trends** — what sits beside it and what opposes it."
        )
        return [Message("system", system), Message("user", user)]
