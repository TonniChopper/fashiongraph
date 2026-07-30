"""Autonomous ReAct agent — decides *when* to reach for a tool.

The static KG/RAG answer what the system already knows; this agent lets the LLM
choose, mid-reasoning, to look something up — the knowledge graph for relational
facts, or the live web for anything current/unknown — then answer with citations.

A text ReAct loop (Thought → Action → Observation → …) is used rather than native
function-calling, so it works on any backend (local Ollama, MLX, API) without
relying on a specific tool-calling schema. Bounded by ``max_steps`` so it can't
run away.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from fg.llm.base import LLM, Message

logger: logging.Logger = logging.getLogger(__name__)

_KNOWN_TOOLS = {"search", "kg", "style", "brand", "trend"}
_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.+?)\]", re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"Final(?:\s*Answer)?:\s*(.+)", re.IGNORECASE | re.DOTALL)

SYSTEM = """You are FashionGraph, an expert fashion stylist and historian agent.

You are the one brain behind a fashion assistant. Decide what the user needs and \
use the right tool. Prefer the knowledge graph for established facts; use web \
search for anything current or unknown.

Available tools (one per Action):

Action: search[<query>]     # live web search — current facts, prices, latest shows
Action: kg[<entity>]        # facts + designer lineage from the knowledge graph
Action: style[<request>]    # personal styling advice (outfits, occasion, capsule)
Action: brand[<brief>]      # define a brand's DNA (aesthetic, values, materials, palette)
Action: trend[<topic>]      # analyse a trend, or rate how plausible/emerging one is

Or finish with:

Final: <your complete answer for the user>

Work one step at a time (Thought → Action → Observation → …). Chain tools when \
useful (e.g. search for current signals, then trend to analyse them). Cite web \
sources inline. Never invent facts — look them up.

If the user has a canvas selection and asks to change or refine it (e.g. "make it \
less formal", "add colour", "darker"), call the SAME tool that produced that card \
with the adjusted request, so an updated card is generated."""


@dataclass
class AgentResult:
    """Outcome of an agent run."""
    answer: str
    sources: list[str] = field(default_factory=list)
    steps: int = 0
    trace: list[str] = field(default_factory=list)
    learned: dict = field(default_factory=dict)
    cards: list[dict] = field(default_factory=list)   # typed artifacts for the canvas


class ReActAgent:
    """LLM + tools (web search, knowledge graph) in a bounded reasoning loop.

    Attributes:
        llm: The language model.
        kg: Optional ``KnowledgeGraph`` for the ``kg`` tool.
        max_steps: Hard cap on tool-use iterations.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        kg: Any | None = None,
        indexer: Any | None = None,
        context_builder: Any | None = None,
        max_steps: int = 4,
        k: int = 4,
        on_step: Callable[[str], None] | None = None,
        on_card: Callable[[dict], None] | None = None,
    ) -> None:
        """Initializes the agent.

        Args:
            llm: LLM backend.
            kg: Optional knowledge graph (enables the ``kg`` tool + ingestion).
            indexer: Optional RAG indexer (enables passage ingestion).
            context_builder: Optional ``ContextBuilder`` — grounds the style/brand/
                trend capability tools in KG + RAG.
            max_steps: Max tool iterations before forcing a final answer.
            k: Web-search results per ``search`` call.
            on_step: Optional callback for live step logging.
        """
        self.llm = llm
        self.kg = kg
        self.indexer = indexer
        self.context_builder = context_builder
        self.max_steps = max_steps
        self.k = k
        self.on_step = on_step
        self.on_card = on_card
        self._collected: list[dict] = []      # web results seen this run (for ingestion)
        self._cards: list[dict] = []          # typed cards produced this run

    def run(self, question: str, context: str = "") -> AgentResult:
        """Answers *question*, using tools as needed.

        Args:
            question: The user's request.
            context: Optional referent context — e.g. the canvas object(s) the user
                selected, so "review this" / "make it less formal" bind to a
                specific thing rather than being ambiguous ("which *this*?").

        Returns:
            An :class:`AgentResult` with answer, cited sources, and a trace.
        """
        user = (f"[Canvas selection — the user is referring to this]\n{context}\n\n{question}"
                if context.strip() else question)
        messages: list[Message] = [Message("system", SYSTEM), Message("user", user)]
        sources: list[str] = []
        trace: list[str] = []
        self._collected = []
        self._cards = []
        self._seen_urls: set[str] = set()

        for step in range(self.max_steps):
            raw = self.llm.chat(messages, temperature=0.2, max_tokens=500)
            final = _FINAL_RE.search(raw)
            action = _ACTION_RE.search(raw)
            known = action and action.group(1).lower() in _KNOWN_TOOLS
            # Prefer a known tool action if one appears before the (maybe hallucinated) Final.
            if known and (not final or action.start() < final.start()):
                tool, arg = action.group(1).lower(), action.group(2).strip()
                obs = self._run_tool(tool, arg, sources)
                trace.append(f"{tool}[{arg}] → {obs[:100]}")
                if self.on_step:
                    self.on_step(f"· {tool}[{arg[:50]}]")
                messages.append(Message("assistant", raw.strip()))
                messages.append(Message("user", f"Observation: {obs}"))
                continue
            if final:
                return AgentResult(final.group(1).strip(), sources, step + 1, trace,
                                   cards=list(self._cards))
            # No parsable action or final → treat the whole reply as the answer.
            return AgentResult(raw.strip(), sources, step + 1, trace, cards=list(self._cards))

        # Out of steps — force a final answer from what we've gathered.
        messages.append(Message("user", "Give your Final answer now, using what you found."))
        raw = self.llm.chat(messages, temperature=0.2, max_tokens=600)
        answer = (_FINAL_RE.search(raw).group(1).strip()
                  if _FINAL_RE.search(raw) else raw.strip())
        return AgentResult(answer, sources, self.max_steps, trace, cards=list(self._cards))

    # ---- tools --------------------------------------------------------

    #: capability tool → card builder in fg.brain.cards
    _CARD_BUILDERS = {"style": "build_style_card", "brand": "build_brand_card",
                      "trend": "build_trend_card"}

    def _run_tool(self, tool: str, arg: str, sources: list[str]) -> str:
        """Executes a tool, records any typed card, returns an observation string."""
        if tool == "search":
            return self._tool_search(arg, sources)
        if tool == "kg":
            return self._tool_kg(arg)
        if tool in self._CARD_BUILDERS:
            return self._capability(tool, arg)
        return f"Unknown tool {tool!r}."

    def _capability(self, tool: str, arg: str) -> str:
        """Runs a capability tool → a typed card (style/brand/trend), grounded in
        KG + RAG. The card goes to the canvas; a short summary feeds the loop."""
        from fg.brain import cards

        builder = getattr(cards, self._CARD_BUILDERS[tool])
        card = builder(self.llm, arg, self.context_builder)
        self._cards.append(card)
        if self.on_card:
            self.on_card(card)
        # Observation for the reasoning loop = the card's human summary.
        return card.get("text") or card.get("title") or card.get("verdict") or "(card produced)"

    def _tool_search(self, query: str, sources: list[str]) -> str:
        from fg.tools.web_search import web_search

        results = web_search(query, self.k)
        if not results:
            return "No web results (or search unavailable)."
        lines: list[str] = []
        for r in results:
            if r["url"]:
                sources.append(r["url"])
                if r["url"] not in self._seen_urls:   # remember for post-answer ingestion
                    self._seen_urls.add(r["url"])
                    self._collected.append(r)
            lines.append(f"- {r['title']}: {r['snippet'][:200]} ({r['url']})")
        return "\n".join(lines)

    # ---- autonomous knowledge growth ---------------------------------

    def ingest_collected(self, *, fetch_pages: bool = True, max_chunks_per_page: int = 2) -> dict:
        """Folds the web results seen this run into the KG (+ RAG).

        Call *after* :meth:`run` so answering stays fast; the agent then grows its
        own knowledge from exactly what it read. No-op if nothing was searched or
        neither a KG nor an indexer is attached.

        Returns:
            Ingestion stats (``{"triples_added","chunks_indexed","sources"}``) or
            ``{}`` if there was nothing to do.
        """
        if not self._collected or (self.kg is None and self.indexer is None):
            return {}
        from fg.tools.web_search import ingest_results

        return ingest_results(
            self._collected, self.llm, kg=self.kg, indexer=self.indexer,
            fetch_pages=fetch_pages, max_chunks_per_page=max_chunks_per_page,
        )

    def _tool_kg(self, entity: str) -> str:
        if self.kg is None:
            return "Knowledge graph unavailable."
        from fg.brain import cards

        card = cards.build_lineage_card(self.kg, entity)
        if card is not None:
            self._cards.append(card)          # deterministic lineage card for the canvas
            if self.on_card:
                self.on_card(card)
        facts = self.kg.facts_as_text(entity, limit=20)
        return "\n".join(f"- {f}" for f in facts) if facts else f"No KG facts for {entity!r}."
