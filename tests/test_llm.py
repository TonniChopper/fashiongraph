"""Tests for the LLM interface (message formatting + factory + backends).

These avoid any real model call by using a fake backend and monkeypatching
the network layer, so they run without torch/ollama/openai installed.
"""

import pytest

from fg.llm.base import LLM, Message
from fg.llm.factory import get_llm
from fg.llm.base import LLMError


def test_message_as_dict():
    assert Message("user", "hi").as_dict() == {"role": "user", "content": "hi"}


class _Echo(LLM):
    """Minimal concrete LLM that echoes the last user message."""

    model = "echo"

    def chat(self, messages, *, temperature=None, max_tokens=None):
        return messages[-1].content


def test_complete_builds_system_and_user():
    """`complete` should prepend a system message when provided."""
    captured = {}

    class _Capture(LLM):
        model = "cap"

        def chat(self, messages, *, temperature=None, max_tokens=None):
            captured["roles"] = [m.role for m in messages]
            return "ok"

    _Capture().complete("q", system="be terse")
    assert captured["roles"] == ["system", "user"]


def test_complete_without_system():
    assert _Echo().complete("hello world") == "hello world"


def test_factory_unknown_backend_raises():
    with pytest.raises(LLMError):
        get_llm("does-not-exist")


def test_factory_builds_ollama_without_calling():
    """Factory should construct the Ollama backend lazily (no network)."""
    llm = get_llm("ollama", model="qwen2.5:7b-instruct", host="http://x:1")
    assert llm.model == "qwen2.5:7b-instruct"


def test_ollama_chat_parses_response(monkeypatch):
    """OllamaLLM.chat should extract message.content from the JSON."""
    from fg.llm import ollama_backend

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "wide-leg denim"}}

    monkeypatch.setattr(
        ollama_backend.requests, "post", lambda *a, **k: _Resp()
    )
    llm = ollama_backend.OllamaLLM(model="m", host="http://x:1")
    out = llm.chat([Message("user", "trend?")])
    assert out == "wide-leg denim"
