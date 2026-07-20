# FashionGraph

Full-cycle AI fashion assistant — an agent brain (router + fusion context +
output contracts) over a clean fashion knowledge core. Three modes: **B2B**,
**Brand Bootstrapper**, **Personal Stylist**.

> Rebuild in progress. See **[REBUILD_PLAN.md](REBUILD_PLAN.md)** for the
> architecture, compute-aware tech decisions, phased roadmap, and the concrete
> techniques adopted from the Farfetch KG paper, FashionKLIP, and the
> ashleyashok reference repo.

## Status — Phase 0 (foundation) complete

- `fg/` package with `config.py` (pydantic-settings, one place for env/paths)
- `fg/llm/` — provider-agnostic LLM interface (Ollama/MLX local, OpenAI API,
  MLX-LoRA later); structured messages, so no model-specific template bugs
- `fg/models/` — CLIP encoder (gradient-flow bug fixed), Temporal GNN
- `fg/rag/` — Chroma indexer (idempotent upsert), retriever (empty-safe),
  visual retriever with **Reciprocal Rank Fusion** dual-path search
- `fg/training/` — CLIP fine-tune with a proper validation split
- `tests/` — unit tests (run `pytest`)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[rag,vision]"   # add RAG + vision extras as needed
pytest
fgraph info                      # show resolved config (command is `fgraph`, not `fg`)
```

Local LLM (Apple Silicon): install [Ollama](https://ollama.com) and
`ollama pull qwen2.5:7b-instruct`. Or set `FG_LLM_BACKEND=openai` with
`OPENAI_API_KEY` in `.env`.

## Layout

```
fg/
├── brain/          # router, context_builder, output_contract, memory (Phase 2)
├── capabilities/   # understand / create / strategize / personal_stylist
├── llm/            # LLM interface + backends
├── models/         # CLIP encoder, Temporal GNN
├── rag/            # indexer, retriever, visual_retriever, fusion (RRF)
├── kg/             # lightweight knowledge graph + canonical vocabulary (Phase 1)
├── data/           # dataset loaders + ingest CLI (Phase 1)
├── training/       # Colab/Kaggle training entry points
└── api/            # FastAPI + WebSocket (Phase 6)
```
