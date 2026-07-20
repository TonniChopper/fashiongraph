"""FashionRouter — classify intent, dispatch to a capability.

Hybrid classifier: fast keyword rules first, optional LLM fallback for
ambiguous queries. This is the "agent-with-tools, not a rigid pipeline" core —
capabilities register against intents and the router decides what to activate.
"""

from __future__ import annotations

import logging
from enum import Enum

from fg.brain.output_contract import OutputContract
from fg.capabilities.base import Capability, CapabilityResult
from fg.llm.base import LLM, Message

logger: logging.Logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Coarse request intents the system can serve."""

    ANALYZE = "analyze"
    DESIGN = "design"
    PATTERN = "pattern"
    BOOTSTRAP = "bootstrap"
    STYLE = "style"
    FULL_CYCLE = "full_cycle"
    UNKNOWN = "unknown"


#: Keyword cues per intent (lowercased substring match).
_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.BOOTSTRAP: (
        "start a brand", "new brand", "brand from scratch", "bootstrap",
        "launch a label", "found a brand", "co-founder", "my own line",
    ),
    Intent.STYLE: (
        "what should i wear", "style me", "outfit", "my wardrobe", "look good",
        "how do i wear", "styling",
    ),
    Intent.ANALYZE: (
        "trend", "forecast", "analyse", "analyze", "what's popular", "brand dna",
        "competitor", "cultural",
    ),
    Intent.DESIGN: ("design", "generate a", "create a piece", "collection", "remix"),
    Intent.PATTERN: ("pattern", "sewing", "tech pack", "garment construction"),
    Intent.FULL_CYCLE: ("end to end", "full cycle", "everything from", "entire process"),
}


class FashionRouter:
    """Routes a natural-language request to a registered capability.

    Attributes:
        llm: Optional LLM for fallback classification.
        capabilities: Registered capabilities keyed by intent.
    """

    def __init__(self, llm: LLM | None = None) -> None:
        """Initializes the router.

        Args:
            llm: Optional LLM used only when keyword rules are inconclusive.
        """
        self.llm = llm
        self.capabilities: dict[Intent, Capability] = {}

    def register(self, capability: Capability) -> None:
        """Registers a capability against each of its declared intents.

        Args:
            capability: The capability to register.
        """
        for intent_name in capability.intents:
            try:
                self.capabilities[Intent(intent_name)] = capability
            except ValueError:
                logger.warning("Capability %s declares unknown intent %r",
                               capability.name, intent_name)

    def classify(self, query: str) -> Intent:
        """Classifies *query* into an :class:`Intent`.

        Keyword rules first; if they tie/miss and an LLM is available, ask it.

        Args:
            query: The user request.

        Returns:
            The best-matching intent (``UNKNOWN`` if nothing fits).
        """
        q = query.lower()
        scores: dict[Intent, int] = {
            intent: sum(1 for kw in kws if kw in q)
            for intent, kws in _KEYWORDS.items()
        }
        best = max(scores, key=lambda i: scores[i])
        if scores[best] > 0:
            return best

        if self.llm is not None:
            return self._llm_classify(query)
        return Intent.UNKNOWN

    def _llm_classify(self, query: str) -> Intent:
        """Uses the LLM to pick an intent label from the fixed set."""
        options = ", ".join(i.value for i in Intent if i != Intent.UNKNOWN)
        prompt = (
            f"Classify this fashion request into exactly one label from "
            f"[{options}]. Reply with only the label.\n\nRequest: {query}"
        )
        try:
            raw = self.llm.chat([Message("user", prompt)], max_tokens=8).strip().lower()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM classification failed (%s).", exc)
            return Intent.UNKNOWN
        for intent in Intent:
            if intent.value in raw:
                return intent
        return Intent.UNKNOWN

    def route(
        self, query: str, contract: OutputContract | None = None
    ) -> CapabilityResult:
        """Classifies then dispatches to the matching capability.

        Args:
            query: The user request.
            contract: Output policy passed through to the capability.

        Returns:
            The capability's result.

        Raises:
            LookupError: If no capability is registered for the intent.
        """
        intent = self.classify(query)
        cap = self.capabilities.get(intent)
        if cap is None:
            raise LookupError(
                f"No capability registered for intent '{intent.value}'. "
                f"Registered: {[i.value for i in self.capabilities]}"
            )
        logger.info("Routed %r → intent=%s → %s", query, intent.value, cap.name)
        return cap.run(query, contract)
