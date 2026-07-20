"""Output contract — controls answer depth and format.

The same capability can speak at different registers: a one-line chat reply, a
detailed report, or an expert breakdown. The contract turns those choices into
a style directive appended to the LLM system prompt, so capabilities don't each
reinvent tone control.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Depth(str, Enum):
    """How much detail to produce."""

    SURFACE = "surface"    # brief, conversational
    DETAILED = "detailed"  # structured, thorough
    EXPERT = "expert"      # deep, technical, with rationale


class Format(str, Enum):
    """The shape of the output."""

    CHAT = "chat"      # prose reply
    REPORT = "report"  # sectioned markdown document
    VISUAL = "visual"  # structured for canvas/cards (JSON-ish)


_DEPTH_DIRECTIVE: dict[Depth, str] = {
    Depth.SURFACE: "Be concise and conversational — a few sentences, no headers.",
    Depth.DETAILED: (
        "Be thorough and well-structured. Use clear sections and concrete, "
        "specific recommendations."
    ),
    Depth.EXPERT: (
        "Give an expert, in-depth treatment: precise fashion terminology, "
        "rationale behind each choice, and trade-offs a creative director "
        "would weigh."
    ),
}

_FORMAT_DIRECTIVE: dict[Format, str] = {
    Format.CHAT: "Respond in natural prose.",
    Format.REPORT: (
        "Respond as a markdown report with '##' section headers and tight "
        "prose under each."
    ),
    Format.VISUAL: (
        "Respond as compact, labelled blocks suitable for cards on a canvas; "
        "keep each block self-contained."
    ),
}


@dataclass(frozen=True)
class OutputContract:
    """Depth + format policy for a response.

    Attributes:
        depth: Level of detail.
        format: Output shape.
    """

    depth: Depth = Depth.DETAILED
    format: Format = Format.REPORT

    def style_directive(self) -> str:
        """Returns the directive text to append to a system prompt."""
        return f"{_DEPTH_DIRECTIVE[self.depth]} {_FORMAT_DIRECTIVE[self.format]}"

    @classmethod
    def from_strings(cls, depth: str = "detailed", format: str = "report") -> "OutputContract":
        """Builds a contract from plain strings (e.g. CLI flags).

        Args:
            depth: One of ``surface|detailed|expert``.
            format: One of ``chat|report|visual``.

        Returns:
            The corresponding ``OutputContract``.

        Raises:
            ValueError: If a value is not recognised.
        """
        try:
            return cls(Depth(depth), Format(format))
        except ValueError as exc:
            raise ValueError(
                f"Bad contract: depth={depth!r}, format={format!r}"
            ) from exc
