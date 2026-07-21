"""Central configuration for FashionGraph.

Loads settings once from environment / ``.env`` via pydantic-settings, so no
module reaches for ``os.environ`` directly. Import the singleton:

    from fg.config import settings
    print(settings.chroma_dir)

Everything is overridable by an env var with the ``FG_`` prefix, e.g.
``FG_LLM_BACKEND=api`` or ``FG_OLLAMA_MODEL=qwen2.5:7b``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (fg/config.py -> repo/)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, populated from env / .env.

    Attributes are grouped by concern: paths, LLM, embeddings, RAG.
    """

    model_config = SettingsConfigDict(
        env_prefix="FG_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Paths --------------------------------------------------------
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    chroma_dir: Path = REPO_ROOT / "data" / "chroma"
    embeddings_dir: Path = REPO_ROOT / "data" / "embeddings"
    checkpoints_dir: Path = REPO_ROOT / "data" / "checkpoints"

    # ---- LLM ----------------------------------------------------------
    #: Which backend the LLM factory builds by default.
    llm_backend: Literal["ollama", "openai", "gemini"] = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512

    # ---- Secrets (loaded from .env; may be empty in dev) --------------
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    wandb_api_key: str = Field(default="", alias="WANDB_API_KEY")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ---- Embeddings ---------------------------------------------------
    #: Default fashion image/text embedder (SOTA on fashion retrieval).
    fashion_embed_model: str = "Marqo/marqo-fashionSigLIP"
    #: Garment segmentation model (segments an outfit photo into pieces).
    seg_model: str = "sayeed99/segformer_b3_clothes"
    #: Filename (under embeddings_dir) of the built product visual index.
    visual_index_name: str = "products_fashionsiglip.npz"
    #: Text embedder for RAG / metadata re-ranking.
    text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Device for text embedding: "auto" (mps→cuda→cpu), or force "mps"/"cpu".
    embed_device: str = "auto"

    # ---- RAG ----------------------------------------------------------
    chroma_collection: str = "fashion_knowledge"
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k: int = 5


#: Import this singleton everywhere.
settings = Settings()
