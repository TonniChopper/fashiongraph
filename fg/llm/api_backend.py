"""OpenAI-compatible API backend.

Used for quality / no-local-setup paths. The ``openai`` package is imported
lazily so the rest of FashionGraph works without it installed.
"""

from __future__ import annotations

import logging

from fg.config import settings
from fg.llm.base import LLM, LLMError, Message

logger: logging.Logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """Chat LLM backed by the OpenAI Chat Completions API.

    Attributes:
        model: OpenAI model name (e.g. ``"gpt-4o-mini"``).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initializes the OpenAI backend.

        Args:
            model: Model name; defaults to ``settings.openai_model``.
            api_key: API key; defaults to ``settings.openai_api_key``.

        Raises:
            LLMError: If the ``openai`` package is missing or no key is set.
        """
        self.model: str = model or settings.openai_model
        key: str = api_key or settings.openai_api_key
        if not key:
            raise LLMError(
                "No OpenAI API key. Set OPENAI_API_KEY in .env or pass api_key."
            )
        try:
            from openai import OpenAI  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "openai package not installed. `pip install openai`."
            ) from exc
        self._client = OpenAI(api_key=key)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Calls the Chat Completions endpoint.

        Args:
            messages: Ordered chat history.
            temperature: Sampling temperature.
            max_tokens: Max new tokens.

        Returns:
            The assistant reply text.

        Raises:
            LLMError: On API errors.
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[m.as_dict() for m in messages],
                temperature=(
                    temperature
                    if temperature is not None
                    else settings.llm_temperature
                ),
                max_tokens=(
                    max_tokens
                    if max_tokens is not None
                    else settings.llm_max_tokens
                ),
            )
        except Exception as exc:  # noqa: BLE001 — surface any client error
            logger.error("OpenAI request failed: %s", exc)
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        return resp.choices[0].message.content or ""
