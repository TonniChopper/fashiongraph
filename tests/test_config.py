"""Tests for the central settings object."""

from pathlib import Path

from fg.config import settings, Settings


def test_settings_singleton_types():
    assert isinstance(settings.chroma_dir, Path)
    assert settings.rag_top_k > 0
    assert settings.fashion_embed_model.startswith("Marqo/")


def test_env_prefix_override(monkeypatch):
    monkeypatch.setenv("FG_OLLAMA_MODEL", "llama3.1:8b")
    s = Settings()
    assert s.ollama_model == "llama3.1:8b"


def test_default_backend_is_ollama():
    assert settings.llm_backend == "ollama"
