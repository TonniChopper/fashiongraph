"""LLM factory — builds the configured backend.

    from fg.llm import get_llm
    llm = get_llm()                 # uses settings.llm_backend
    llm = get_llm("openai")         # explicit
    print(llm.complete("Name three fall 2026 silhouettes."))
"""

from __future__ import annotations

from fg.config import settings
from fg.llm.base import LLM, LLMError


def get_llm(backend: str | None = None, vision: bool = False, **kwargs) -> LLM:
    """Instantiates an :class:`~fg.llm.base.LLM` for the given backend.

    Args:
        backend: ``"ollama"``, ``"openai"``, or ``"gemini"``. Defaults to
            ``settings.llm_backend``.
        vision: If ``True``, select an image-capable model (Ollama vision model;
            OpenAI is already multimodal).
        **kwargs: Passed through to the backend constructor (e.g. ``model``).

    Returns:
        A ready-to-use LLM instance.

    Raises:
        LLMError: If the backend name is unknown.
    """
    name: str = (backend or settings.llm_backend).lower()

    if name == "ollama":
        from fg.llm.ollama_backend import OllamaLLM

        if vision and "model" not in kwargs:
            kwargs["model"] = settings.ollama_vision_model
        return OllamaLLM(**kwargs)
    if name in {"openai", "api"}:
        from fg.llm.api_backend import OpenAILLM

        return OpenAILLM(**kwargs)
    if name == "gemini":  # pragma: no cover — wired in a later phase
        raise LLMError("Gemini backend not implemented yet.")

    raise LLMError(f"Unknown LLM backend: {name!r}")
