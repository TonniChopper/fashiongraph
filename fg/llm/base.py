"""LLM interface and shared message types.

The key design lesson from the old ``fashion_llm.py``: never hand-write a
model-specific chat template in application code (it emitted LLaMA control
tokens into a Mistral model). Instead we pass structured ``Message`` objects
and let each backend apply its own correct template.
"""

from __future__ import annotations

import abc
import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


def encode_image(image: Any) -> str:
    """Encodes an image to base64 (for multimodal messages).

    Args:
        image: A file path, raw bytes, or a PIL image.

    Returns:
        Base64-encoded JPEG string (no data-URI prefix).
    """
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    elif isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    else:  # assume PIL image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        data = buf.getvalue()
    return base64.b64encode(data).decode("ascii")


@dataclass
class Message:
    """A single chat message, optionally multimodal.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``.
        content: The message text.
        images: Base64-encoded images attached to this message (for VLMs).
    """

    role: Role
    content: str
    images: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Returns the Ollama-style message dict (includes ``images`` if any)."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d


class LLM(abc.ABC):
    """Abstract chat LLM. Backends implement :meth:`chat`.

    Attributes:
        model: The backing model identifier.
    """

    model: str

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generates an assistant reply for a message list.

        Args:
            messages: Ordered chat history (system first, typically).
            temperature: Sampling temperature; falls back to config default.
            max_tokens: Max new tokens; falls back to config default.

        Returns:
            The assistant's reply text.

        Raises:
            LLMError: If the backend fails.
        """
        raise NotImplementedError

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Convenience wrapper: single-turn completion.

        Args:
            prompt: The user prompt.
            system: Optional system instruction.
            temperature: Sampling temperature.
            max_tokens: Max new tokens.

        Returns:
            The assistant's reply text.
        """
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )


class LLMError(RuntimeError):
    """Raised when an LLM backend call fails."""
