"""Ollama backend — local LLM on Apple Silicon (MLX-accelerated).

Talks to a running Ollama server over HTTP. Ollama applies the correct chat
template for whichever model is pulled, so we just send structured messages.

    ollama pull qwen2.5:7b-instruct
    ollama serve   # usually already running
"""

from __future__ import annotations

import logging

import requests

from fg.config import settings
from fg.llm.base import LLM, LLMError, Message

logger: logging.Logger = logging.getLogger(__name__)


class OllamaLLM(LLM):
    """Chat LLM backed by a local Ollama server.

    Attributes:
        model: Ollama model tag (e.g. ``"qwen2.5:7b-instruct"``).
        host: Base URL of the Ollama server.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initializes the Ollama backend.

        Args:
            model: Model tag; defaults to ``settings.ollama_model``.
            host: Server URL; defaults to ``settings.ollama_host``.
            timeout: Per-request timeout in seconds.
        """
        self.model: str = model or settings.ollama_model
        self.host: str = (host or settings.ollama_host).rstrip("/")
        self.timeout: float = timeout

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Sends a chat request to ``/api/chat`` and returns the reply.

        Args:
            messages: Ordered chat history.
            temperature: Sampling temperature.
            max_tokens: Max new tokens (mapped to Ollama ``num_predict``).

        Returns:
            The assistant reply text.

        Raises:
            LLMError: On connection errors or non-200 responses.
        """
        payload: dict = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": (
                    temperature
                    if temperature is not None
                    else settings.llm_temperature
                ),
                "num_predict": (
                    max_tokens
                    if max_tokens is not None
                    else settings.llm_max_tokens
                ),
            },
        }
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise LLMError(
                f"Ollama request to {self.host} failed: {exc}"
            ) from exc

        data: dict = resp.json()
        return data.get("message", {}).get("content", "")
