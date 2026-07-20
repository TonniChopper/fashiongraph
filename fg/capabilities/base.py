"""Capability protocol — the unit the router dispatches to.

A capability is a self-contained skill (bootstrap a brand, review a look,
analyse a trend). It receives a request + an output contract and returns a
result. Capabilities pull their own grounding via a ``ContextBuilder``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from fg.brain.output_contract import OutputContract


@dataclass
class CapabilityResult:
    """The output of running a capability.

    Attributes:
        text: The human-facing answer (markdown/prose).
        data: Optional structured payload (for canvas/API consumers).
        sources: Source tags used for grounding, for citation.
    """

    text: str
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


class Capability(abc.ABC):
    """Base class for all capabilities.

    Attributes:
        name: Stable identifier.
        intents: Router intents this capability serves.
    """

    name: str = "capability"
    intents: tuple[str, ...] = ()

    @abc.abstractmethod
    def run(
        self, request: Any, contract: OutputContract | None = None
    ) -> CapabilityResult:
        """Executes the capability.

        Args:
            request: Capability-specific input (a query string, an answers
                dict, an image handle, …).
            contract: Output depth/format policy; a sensible default is used
                if ``None``.

        Returns:
            A :class:`CapabilityResult`.
        """
        raise NotImplementedError
