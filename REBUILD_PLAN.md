# FashionGraph — Rebuild Plan

> Clean-slate rebuild. Salvage the strong model/RAG code, throw away the current data
> and the script sprawl, and build the system the design doc actually describes:
> an agent brain (router + fusion + output contracts) over a small, *clean* knowledge core.
>
> Constraints that shape everything: dev machine is a **MacBook Air M4, 24 GB (MPS, no CUDA)**;
> heavy training is **free Colab / Kaggle** in bursts. No deadline — quality over speed.

---

## 0. Guiding principles

1. **Quality core, not big core.** The current data is high-volume, low-signal (raw Wikipedia dumps + noisy scrapes). We replace it with a *small, curated, well-tagged* corpus. 500 excellent chunks beat 50k mediocre ones for RAG.
2. **Everything behind an interface.** LLM, embedder, retriever, and each capability sit behind a stable Python protocol so we can swap Ollama ↔ API ↔ fine-tuned model without touching callers.
3. **Compute-aware ordering.** GPU-free work first (data, brain, RAG, Bootstrapper). Inference-only ML next (pretrained fashion embeddings run fine on MPS). Training (GNN, LoRA) deferred to Colab/Kaggle sessions. Generative diffusion is dead last.
4. **Every phase ships something demoable.** No phase is pure plumbing.
5. **The brain is the product.** The router + context_builder + output_contract is what makes this "one assistant" instead of a script folder. Build it early, even if thin.

---

## 1. Salvage decisions — keep / rewrite / kill

**Keep (mostly as-is, minor fixes):**
- `src/models/clip_encoder.py` — solid. Fix the `torch.no_grad()` bug so the unfrozen blocks can actually train. But see §2: default the embedder to **Marqo-FashionSigLIP**, keep this as the fine-tune path.
- `src/models/temporal_gnn.py` — clean IL + GTAM implementation. Keep; feed it *real* trend data later.
- `src/rag/retriever.py` + `indexer.py` — good bones. Switch `.add` → `upsert`, add empty-collection guard.
- `src/rag/visual_retriever.py` — well-built; rewire to the new embedder.
- Training loop `train_clip.py` — keep, add a validation split.

**Rewrite:**
- `src/models/fashion_llm.py` — **hard rewrite.** Kill the `bitsandbytes` 4-bit + `device_map="auto"` path (won't run on M4). Replace with an `LLM` interface backed by Ollama (MLX) locally / API remotely / MLX-LoRA when fine-tuned. Fix the prompt template (currently emits LLaMA-3.1 tokens into a Mistral model).
- Data pipeline — all of `src/scripts/*` collapses into one small, config-driven ingest CLI.

**Kill:**
- All current scraped data (`data/raw/*`, `data/embeddings/*`, both `chroma_db/` and `data/chroma/`).
- Script sprawl: `build_index`, `build_faiss_index`, `build_fashion_index`, `index_articles`, `index_runway`, `embed_runway`, `vogue.py` vs `fetch_vogue_runway.py` → merge into `python -m fg.data ...`.
- `torch_geometric_temporal` dependency (GNN is hand-rolled).
- Committed binaries (41 MB of Chroma in git) + the tracked pickle. Fix `.gitignore` (currently every line is duplicated).

---

## 2. Compute-aware tech stack (the decisions that matter)

| Layer | Decision | Why (given M4 + free GPU) |
|---|---|---|
| **Local LLM inference** | **Ollama** (MLX-backed) for dev; small models (Qwen/Llama 3.x 7–8B, Q4) | Runs comfortably in 24 GB; no CUDA needed |
| **LLM fine-tune** | **MLX-LM LoRA** on the Mac for 7–8B; Colab/Kaggle for anything bigger | MLX is the *only* local LoRA path on Apple Silicon; 24 GB fits 7–8B |
| **LLM fallback / quality** | OpenAI / Gemini API behind the same interface | For the Bootstrapper MVP, an API model gets us shipping without any local pain |
| **Fashion embeddings** | **Marqo-FashionSigLIP** (pretrained) as default; our fine-tuned CLIP as an optional path | SOTA on fashion retrieval (+57% recall@1 vs FashionCLIP 2.0); inference-only, runs on MPS |
| **Text embeddings (RAG)** | `bge-small`/`all-MiniLM` via sentence-transformers, or Chroma default | Cheap, CPU-fine |
| **Vector store** | ChromaDB (persistent), one store, gitignored | Already integrated |
| **GNN training** | Colab/Kaggle (T4/P100) in checkpointed bursts | Tiny model, short runs |
| **Diffusion (Phase 7)** | Colab Pro / RunPod A100 only | Far too heavy for M4 |

**Rule:** nothing in the critical path depends on CUDA-only libs. `bitsandbytes` is out.

---

## 3. Target repo structure (the brain the doc promises)

```
fashiongraph/
├── fg/                          # single importable package (replaces loose src/)
│   ├── brain/
│   │   ├── router.py            # FashionRouter: intent classification → capabilities
│   │   ├── context_builder.py   # Fusion Context: assembles RAG + visual + trend + DNA
│   │   ├── output_contract.py   # surface/detailed/expert × chat/report/visual
│   │   └── memory.py            # session / brand / user memory
│   ├── capabilities/
│   │   ├── base.py              # Capability protocol (name, intents, run(ctx))
│   │   ├── understand/          # trend analysis, look analysis, brand DNA
│   │   ├── create/              # styling, moodboard→brief, design gen (later)
│   │   ├── strategize/          # bootstrapper, demand, content
│   │   └── personal_stylist/    # look review, lookbook, wardrobe memory
│   ├── models/                  # clip_encoder, temporal_gnn, dna_encoder
│   ├── llm/                     # LLM interface + ollama / api / mlx backends
│   ├── rag/                     # indexer, retriever, visual_retriever
│   ├── data/                    # ingest CLI, cleaners, NER+EL, dataset loaders
│   ├── kg/                      # lightweight knowledge graph (entities, linking)
│   ├── api/                     # FastAPI + WebSocket (later phases)
│   └── config.py                # pydantic-settings, loads .env once
├── data/                        # gitignored: raw/, processed/, chroma/, embeddings/
├── tests/                       # actually populated this time
├── notebooks/                   # Colab/Kaggle training notebooks
├── frontend/                    # React + tldraw (later phases)
└── pyproject.toml               # replaces bare requirements.txt
```

---

## 4. Data strategy v2 — quality-first (the big fix)

The current corpus is the weakest link. New approach:

**Sources (curated, not scraped-in-bulk):**
- **Structured product data:** H&M Personalized Fashion, DeepFashion2, Polyvore (outfit graphs), Marqo fashion datasets — for attributes, categories, outfit compatibility.
- **Editorial / trend text:** a *small hand-picked* set of runway reviews and trend pieces, cleaned — not the whole of Vogue.
- **Reference knowledge:** curated brand/DNA facts, not raw Wikipedia articles.

**Pipeline (one CLI, `python -m fg.data`):**
1. **Ingest** → normalized records with mandatory provenance (`source`, `date`, `type`, `brand`, `season`).
2. **Clean & dedup** → strip boilerplate, de-duplicate near-identical chunks, drop junk.
3. **NER + Entity Linking** (Farfetch method: `arxiv 2206.01087`) → extract fashion entities (garment, silhouette, material, color, brand, era) and link to a **canonical fashion vocabulary**.
4. **KG-lite** → store entities + relations (`kg/`); this is the seed of the Trend Genealogy Tree and Runway Archive queries.
5. **Chunk + embed** → semantic chunks, each carrying its entity tags + metadata, into ChromaDB.

**Quality bar:** every chunk is traceable (source + date), tagged with linked entities, and deduped. Target a few thousand *clean* chunks before scaling.

---

## 5. Phased roadmap

Each phase is independently demoable. GPU need noted per phase.

### Phase 0 — Foundation reset · *no GPU*
Repo restructure to `fg/`, `pyproject.toml`, `config.py` (loads `.env` once), fixed `.gitignore`, purge committed binaries, wire CI + a first `tests/` skeleton. Fix the CLIP `no_grad` bug and RAG `upsert` while we're in there.
**Ships:** clean repo that installs and imports; green test run.

### Phase 1 — Data pipeline v2 + KG-lite · *no GPU*
Build the ingest CLI, cleaners, NER+EL, canonical vocabulary, KG-lite, and the ChromaDB index from a curated seed set.
**Ships:** `fg.data build` produces a clean, tagged, queryable knowledge base; a retrieval smoke-test that returns sensible, sourced chunks.

### Phase 2 — Brain skeleton + Brand Bootstrapper MVP · *no GPU*
`router.py` (intent classification, start rule/LLM hybrid), `context_builder.py`, `output_contract.py`, `memory.py`, the `LLM` interface (Ollama local + API fallback), and the first capability: **Brand Bootstrapper** (10-question brand-from-scratch → DNA + strategy + starter collection brief), grounded in RAG.
**Ships:** end-to-end CLI/notebook demo: answer 10 questions → get a coherent brand. First real showcase.

### Phase 3 — Visual layer + Personal Stylist look review · *MPS inference*
Integrate **Marqo-FashionSigLIP**, rebuild the visual retriever on it, add **Look Analysis** (silhouette / palette / occasion) and **Personal Stylist look review** as capabilities. Wire visual signals into `context_builder`.
**Ships:** photo in → structured look review + "similar runway looks" out.

### Phase 4 — Temporal trends + Genealogy · *Colab/Kaggle training*
Build real seasonal trend data from the KG, train the Temporal GNN, expose trend scores to the brain, first cut of the Trend Genealogy Tree and Runway Archive queries.
**Ships:** "what's rising for next season" + trend lineage view.

### Phase 5 — Fashion LLM fine-tune · *MLX on Mac / Colab*
Generate a clean instruction dataset in fashion voice, LoRA fine-tune a 7–8B model via **MLX-LM** (knowledge-injection flavored, FashionKLIP-style), serve it through the existing `LLM` interface.
**Ships:** noticeably more fluent, on-domain assistant with no code changes upstream.

### Phase 6 — API + Canvas · *no GPU*
FastAPI endpoints per capability, WebSocket layer, **tldraw** canvas with `@tldraw/sync`, `ai_bridge` mapping canvas events → router intents, React front end (Bootstrapper, Trend Explorer, MoodBoard, Personal Stylist views).
**Ships:** the interactive infinite-canvas product.

### Phase 7 — Generative modules · *Colab Pro / RunPod A100*
Design + pattern generation (SDXL + ControlNet + IP-Adapter, GarmentDiffusion), Collection Builder with Coherence Score, advanced scores (Viral, Conflict). Most expensive, intentionally last.
**Ships:** generate-a-collection demo for portfolio/pitch.

---

## 6. Suggested first sprint (concrete)

1. Scaffold `fg/` + `pyproject.toml` + `config.py` + fixed `.gitignore`; purge committed binaries and old data.
2. Move & fix the keeper modules (`clip_encoder` no_grad fix, `retriever` upsert/guard).
3. Write the `LLM` interface + Ollama backend + one API backend; delete the bitsandbytes path.
4. Stand up `tests/` with unit tests for the loss, GNN shapes, and retriever contract.
5. Define the curated data seed list and the canonical fashion vocabulary schema.

That gets us to a clean, installable, tested foundation with a working LLM interface — the launchpad for the Bootstrapper MVP in Phase 2.

---

## 7. Reference approaches to copy (verified from source)

Concrete, copyable techniques from the papers/repos in the brief — what to lift from each.

### Farfetch KG paper (`arxiv 2206.01087`) → data/KG layer
- **Transfer-learning NER on a tiny labeled set.** They hit **89.75% NER accuracy** by fine-tuning a *pretrained* model on a small hand-labeled set. Takeaway: we only need to hand-label a few hundred product descriptions, not thousands. (Or bootstrap labels with an LLM, then correct.)
- **Extract → normalize → link.** NER pulls attributes; a *simple rule-based / light-ML* Entity Linking step maps them to canonical entities. EL does **not** need a big trained model.
- **Fashion needs its own ontology.** Generic KGs (Wikidata/Yago) don't fit — attributes are too domain-specific. Validates our `kg/` canonical fashion vocabulary as a first-class artifact.

### FashionKLIP (ACL 2023) → visual alignment / CLIP fine-tune
- **Concept-level alignment, not just instance-level.** Build a multi-modal *conceptual* KG (fashion concepts like "barrel-leg denim", "peplum") and align image embeddings to **concept prototypes**, not just individual captions.
- **Copy for Phase 5:** derive concept prototypes from our KG, add a concept-alignment loss on top of the standard contrastive loss when fine-tuning the embedder. This is the "knowledge injection" — cheaper and more robust than plain (image, caption) fine-tuning.

### ashleyashok/fashion-knowledge-graph → end-to-end system blueprint (closest to us)
This repo is effectively a working half of our system. Directly copyable patterns:
- **Segment before embedding.** Uses `sayeed99/segformer_b3_clothes` to cut a photo into garment regions, then embeds each item separately. Adopt for **Look Analysis / Personal Stylist** — turns one outfit photo into per-item nodes.
- **Confirms our embedder:** they use `Marqo/marqo-fashionCLIP` + `all-MiniLM-L6-v2` for text. Independent validation of §2.
- **LLM-powered attribute extraction (GPT-4 + structured prompts)** as an alternative to training NER — simpler for a solo dev. Good default; swap to trained NER (Farfetch method) later if cost/latency matters.
- **KG schema:** Product nodes + attribute nodes + **co-occurrence edges** ("worn together", mined from real outfit images). This *is* our Collection Builder / styling-suggestions substrate.
- **Dual-path search + Reciprocal Rank Fusion (RRF).** Run text-path and visual-path retrieval separately, fuse with RRF. **Replace the ad-hoc weighted blend** currently in `visual_retriever.py` (`0.8*clip + 0.2*text`) with RRF — more robust, no magic weights.
- **"Complete the Look" = graph traversal** over co-occurrence edges. Maps straight onto our styling / collection capabilities.
- **Class-based structure** (`base_model.py` abstractions, `model_manager` for lifecycle, `config/settings.py`) — mirrors our `fg/` layout; worth reading their file organization.
- Note their stack uses Neo4j + Pinecone; we stay on **ChromaDB** (+ a lightweight graph, NetworkX or SQLite, before committing to Neo4j) to stay free/local on the M4.

### FashionCLIP (patrickjohncyh) → embedder baseline
- **Base model matters:** FashionCLIP 2.0 switched to `laion/CLIP-ViT-B-32-laion2B` because it beats OpenAI CLIP on fashion (weighted-F1 0.83 vs 0.66 on FMNIST). Our current code uses `ViT-L-14/openai` — a weaker base for this domain. We supersede it with Marqo anyway, but keep FashionCLIP 2.0 as a reference/fallback.
- **Dead-simple API** (`pip install fashion-clip` → `encode_images`/`encode_text` → normalize → dot product) — use as a quick baseline to benchmark Marqo against before committing.

**Net effect on the plan:** Phase 1 gains LLM-based attribute extraction + co-occurrence KG schema; Phase 3 gains garment segmentation + RRF dual-path search; Phase 5 gains concept-level alignment. Nothing here requires more compute than we have.

---

*This plan supersedes the aspirational structure in the project brief: same vision, sequenced for an M4 + free-GPU reality, quality-core-first.*
