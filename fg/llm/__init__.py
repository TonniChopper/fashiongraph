"""LLM abstraction layer.

One interface, swappable backends (Ollama for local dev on Apple Silicon,
OpenAI/Gemini API for quality, MLX-LoRA later). Callers depend only on the
``LLM`` protocol and never on a specific provider.
"""

from fg.llm.base import LLM, Message
from fg.llm.factory import get_llm

__all__ = ["LLM", "Message", "get_llm"]
