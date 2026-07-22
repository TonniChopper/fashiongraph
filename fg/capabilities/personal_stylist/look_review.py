"""Personal Stylist — Look Review.

Photo in → structured styling review out. Pipeline: segment the outfit into
garments, embed the look with the fashion embedder, retrieve visually similar
catalog pieces, pull styling knowledge from RAG, then have the LLM synthesise a
review (silhouette, palette, occasion-fit, and concrete suggestions).

Every perception component is optional and injected, so the capability degrades
gracefully (and is testable with fakes): no segmenter → skip garments; no index
→ skip visual references; it still writes a useful RAG-grounded review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fg.brain.context_builder import ContextBuilder, FusionContext
from fg.brain.output_contract import Depth, Format, OutputContract
from fg.capabilities._prompts import GROUNDING_DISCIPLINE, STYLING_RUBRIC
from fg.capabilities.base import Capability, CapabilityResult
from fg.llm.base import LLM, Message

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Perception:
    """What the vision stack understood about a look.

    Attributes:
        garments: Detected garment labels (may be empty).
        similar: Visually similar catalog hits (may be empty).
        aesthetic_score: Learned aesthetic score 0–100 (``None`` if no scorer).
    """

    garments: list[str] = field(default_factory=list)
    similar: list[dict[str, Any]] = field(default_factory=list)
    aesthetic_score: int | None = None
    movements: list[tuple[str, float]] = field(default_factory=list)

    def movements_text(self) -> str:
        """Renders the nearest aesthetic movements."""
        return ", ".join(f"{n} ({s})" for n, s in self.movements) if self.movements else ""

    def garments_text(self) -> str:
        """Human-readable garment list."""
        return ", ".join(self.garments) if self.garments else "(not detected)"

    def similar_text(self, limit: int = 5) -> str:
        """Renders similar catalog items as labelled lines."""
        lines: list[str] = []
        for hit in self.similar[:limit]:
            title = hit.get("title") or hit.get("category", "item")
            colour = hit.get("colour", "")
            score = hit.get("score", 0.0)
            lines.append(f"- {title} ({colour}) · sim={score}")
        return "\n".join(lines) if lines else "(no visual references)"


class LookReview(Capability):
    """Reviews an outfit photo and gives styling guidance.

    Attributes:
        name: Capability id.
        intents: Router intents served.
        llm: Language model backend.
        embedder: Optional fashion image/text embedder.
        segmenter: Optional garment segmenter.
        visual_index: Optional visual index for similar-look retrieval.
        context_builder: Supplies RAG styling knowledge.
    """

    name = "look_review"
    intents = ("style",)

    def __init__(
        self,
        llm: LLM,
        *,
        embedder: Any | None = None,
        segmenter: Any | None = None,
        visual_index: Any | None = None,
        aesthetic_scorer: Any | None = None,
        movement_matcher: Any | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        """Initializes the look-review capability.

        Args:
            llm: LLM backend.
            embedder: Optional ``FashionEmbedder``.
            segmenter: Optional ``GarmentSegmenter``.
            visual_index: Optional ``VisualIndex``.
            aesthetic_scorer: Optional ``AestheticScorer`` (learned taste signal).
            movement_matcher: Optional ``MovementMatcher`` (art/architecture lineage).
            context_builder: Optional RAG context builder.
        """
        self.llm = llm
        self.embedder = embedder
        self.segmenter = segmenter
        self.visual_index = visual_index
        self.aesthetic_scorer = aesthetic_scorer
        self.movement_matcher = movement_matcher
        self.context_builder = context_builder or ContextBuilder(None)

    def run(
        self, request: Any, contract: OutputContract | None = None
    ) -> CapabilityResult:
        """Reviews the look at the given image path.

        Args:
            request: Image path ``str`` or dict ``{"image_path", "occasion"}``.
            contract: Output policy; defaults to detailed report.

        Returns:
            A :class:`CapabilityResult` with the styling review.

        Raises:
            FileNotFoundError: If the image can't be opened.
        """
        image_path, occasion = self._parse_request(request)
        image = self._load_image(image_path)
        return self.review(image, occasion=occasion, contract=contract)

    def review(
        self, image: Any, *, occasion: str = "", contract: OutputContract | None = None
    ) -> CapabilityResult:
        """Reviews an already-loaded image (testable seam).

        Args:
            image: A PIL image (or any object the injected components accept).
            occasion: Optional occasion/context (e.g. "job interview").
            contract: Output policy.

        Returns:
            The styling review result.
        """
        contract = contract or OutputContract(Depth.DETAILED, Format.REPORT)
        perception = self._perceive(image)
        ctx = self.context_builder.build(
            self._retrieval_query(perception, occasion), n_rag=5
        )
        messages = self._build_messages(perception, occasion, ctx, contract)
        text = self.llm.chat(messages, max_tokens=900)

        source_set = {h.get("title", "") for h in perception.similar if h.get("title")} | {
            (c.get("metadata", {}).get("title")
             or c.get("metadata", {}).get("source", ""))
            for c in ctx.rag_chunks
        }
        source_set.discard("")
        return CapabilityResult(
            text=text,
            data={"garments": perception.garments, "occasion": occasion},
            sources=sorted(source_set),
        )

    # ---- perception ---------------------------------------------------

    def _perceive(self, image: Any) -> Perception:
        """Runs the available vision components over *image*.

        Embeds the look once and reuses that vector for both similar-look
        retrieval and aesthetic scoring.
        """
        garments: list[str] = []
        similar: list[dict] = []
        score: int | None = None

        if self.segmenter is not None:
            try:
                garments = self.segmenter.labels(image)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Segmentation failed (%s).", exc)

        vec = None
        if self.embedder is not None:
            try:
                vec = self.embedder.encode_images([image])[0]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding failed (%s).", exc)

        if vec is not None and self.visual_index is not None:
            try:
                similar = self.visual_index.search(vec, top_k=5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Visual retrieval failed (%s).", exc)

        if vec is not None and self.aesthetic_scorer is not None:
            try:
                score = self.aesthetic_scorer.score_100(vec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Aesthetic scoring failed (%s).", exc)

        movements: list[tuple[str, float]] = []
        if vec is not None and self.movement_matcher is not None:
            try:
                movements = self.movement_matcher.match(vec, top_k=3)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Movement matching failed (%s).", exc)

        return Perception(
            garments=garments, similar=similar,
            aesthetic_score=score, movements=movements,
        )

    @staticmethod
    def _retrieval_query(perception: Perception, occasion: str) -> str:
        """Builds a styling-knowledge query from garments + occasion."""
        parts = list(perception.garments)
        if occasion:
            parts.append(occasion)
        parts.append("styling advice outfit")
        return " ".join(parts)

    # ---- prompt -------------------------------------------------------

    def _build_messages(
        self,
        perception: Perception,
        occasion: str,
        ctx: FusionContext,
        contract: OutputContract,
    ) -> list[Message]:
        """Assembles the chat messages for the review."""
        system = (
            "You are FashionGraph's Personal Stylist — a warm, sharp-eyed stylist "
            "with real taste, who reviews an outfit honestly and constructively. "
            "Read the detected garments, the similar catalog references, and the "
            "aesthetic score as evidence. Be encouraging but specific; never "
            "body-shame. " + STYLING_RUBRIC + " " + contract.style_directive()
            + " " + GROUNDING_DISCIPLINE
        )
        occasion_line = f"Occasion / context: {occasion}\n" if occasion else ""
        movement_line = (
            f"Aesthetic lineage (nearest art/architecture movements, evocative not "
            f"literal): {perception.movements_text()}. Use these as inspiration to "
            f"describe the look's design language — don't force them.\n"
            if perception.movements else ""
        )
        score_line = (
            f"Aesthetic model score (learned from human preference judgments): "
            f"{perception.aesthetic_score}/100. Treat this as one signal — explain "
            f"what is driving it, don't just repeat the number.\n"
            if perception.aesthetic_score is not None else ""
        )
        user = (
            f"## The look\n"
            f"Detected garments: {perception.garments_text()}\n"
            f"{score_line}{movement_line}{occasion_line}\n"
            f"## Visually similar catalog pieces\n{perception.similar_text()}\n\n"
            f"## Styling knowledge (for grounding)\n"
            f"{ctx.knowledge_block() or '(none retrieved)'}\n\n"
            "## Task\nReview this outfit. Produce these sections:\n"
            "1. **Silhouette & proportion** — how the shapes read together.\n"
            "2. **Colour & palette** — what the palette is doing and whether it works.\n"
            "3. **Occasion fit** — is it right for the context (if given)?\n"
            "4. **What's working** — the strongest elements.\n"
            "5. **Styling moves** — 3 concrete, specific changes or additions to "
            "elevate the look.\n"
            "If garments weren't detected, review from the references and any "
            "occasion given, and say what you'd want to see."
        )
        return [Message("system", system), Message("user", user)]

    # ---- io -----------------------------------------------------------

    @staticmethod
    def _parse_request(request: Any) -> tuple[str, str]:
        """Splits a request into (image_path, occasion)."""
        if isinstance(request, dict):
            return str(request.get("image_path", "")), str(request.get("occasion", ""))
        return str(request), ""

    @staticmethod
    def _load_image(image_path: str) -> Any:
        """Loads a PIL image from *image_path*.

        Raises:
            FileNotFoundError: If the path is empty or unreadable.
        """
        if not image_path:
            raise FileNotFoundError("No image path provided.")
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow required: pip install Pillow") from exc
        return Image.open(image_path).convert("RGB")
