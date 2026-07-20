"""LLM interface and shared message types.

The key design lesson from the old ``fashion_llm.py``: never hand-write a
model-specific chat template in application code (it emitted LLaMA control
tokens into a Mistral model). Instead we pass structured ``Message`` objects
and let each backend apply its own correct template.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """A single chat message.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``.
        content: The message text.
    """

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        """Returns the OpenAI/Ollama-style ``{"role", "content"}`` dict."""
        return {"role": self.role, "content": self.content}


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
