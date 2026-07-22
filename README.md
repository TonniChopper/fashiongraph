# FashionGraph

A full-cycle AI fashion assistant built around a **knowledge-graph-centered**
approach to fashion understanding — an agent brain (intent router + fusion
context + output contracts) over a clean knowledge core, a multimodal vision
layer with a *learned* sense of taste, and a fashion knowledge graph that lets
the system **reason over relationships**, not just retrieve passages.

Master's-thesis project. Built and tested on a MacBook M4 (24 GB) + free Colab —
proven-architecture-first, quality over scale.

> See **[REBUILD_PLAN.md](REBUILD_PLAN.md)** for the architecture, compute-aware
> decisions, phased roadmap, and the techniques adopted from the Farfetch KG
> paper, FashionKLIP, and the ashleyashok reference repo. See
> **[data/DATASETS.md](data/DATASETS.md)** for the dataset choices.

## Status — Phases 0–4 complete (102 tests passing)

**Phase 0 — Foundation.** Clean `fg/` package, `config.py` (one place for
env/paths), provider-agnostic `fg/llm/` (Ollama/MLX local + OpenAI API; structured
messages, so no model-specific template bugs).

**Phase 1 — Knowledge core.** A quality-first data pipeline (`fg/data/`,
`fgraph data`) — clean, dedup, chunk, embed into ChromaDB. ~60k grounded chunks
from curated Wikipedia fashion + product attributes + styling examples. Text
embeddings run on the M4 GPU (MPS).

**Phase 2 — Agent brain.** `fg/brain/` — `FashionRouter` (intent classification →
capability dispatch), `context_builder` (Fusion Context), `output_contract`
(depth × format), `memory`. Three working, RAG-grounded capabilities:
- **Brand Bootstrapper** — 10 questions → Brand DNA, strategy, starter collection.
- **Trend Analysis** — cultural-forecaster read on a trend/era/aesthetic.
- **Personal Stylist look-review** — photo in → structured styling review.

All capabilities share a **grounding discipline** that curbs brand hallucination.

**Phase 3 — Vision + taste.** `fg/vision/` — Marqo-FashionSigLIP embeddings,
garment segmentation (`segformer_b3_clothes`), a numpy visual index, and a
**learned aesthetic scorer** trained on human pairwise preference judgments
(**0.703 held-out pairwise accuracy** on worn-outfit taste). Plus a styling
rubric and an art/architecture **aesthetic-lineage** matcher. The Personal
Stylist now runs on a **vision-language model** (Ollama Qwen2.5-VL / Llama-3.2-
Vision), so it actually *sees* the outfit rather than guessing from labels.

**Phase 4 — Knowledge graph.** `fg/kg/` — LLM-assisted triple extraction into a
fixed fashion ontology, a SQLite triple store, **entity resolution** (alias
merge + noise filter), graph reasoning (path-finding, multi-hop), one-shot-ICL
**link prediction**, and KG facts wired into retrieval. From 100 Wikipedia pages:
**~2,500 triples / ~2,300 entities**.

**Phase 5 (in progress) — Multimodal KG grounding (the novel core).** A
**runway visual index** over ~2,200 designer-labeled Vogue looks (11 houses)
links a photo to the *nearest real runway collections* (image↔image) and
traverses the KG for lineage — turning "reads minimalist" into "nearest to
Marni / Rick Owens; lineage traces to …". See **[NOVEL_IDEAS.md](NOVEL_IDEAS.md)**
for the full multimodal-KG roadmap (visual node prototypes, VLM-extracted edges,
cross-modal alignment).

### Headline result: does the KG earn its place?

Measured KG-grounded answers vs flat RAG on relational questions (8 top entities):

| Measure | KG | flat RAG |
|---|---:|---:|
| Mean relational-fact coverage | **0.38–0.40** | 0.13–0.14 |
| → lift | **+0.24 to +0.26** | — |
| Independent LLM-judge (wins) | **5** | 3 (0 ties) |

Coverage favors KG on **8/8 entities**, reproducibly. The independent judge
*modestly* favors KG (5–3) — and where the two measures disagree (KG has more
facts but the judge prefers RAG), we learn that **fact density ≠ answer quality**:
KG facts must be *synthesized*, not dumped, and extraction noise affects quality.
And `kg path` / `kg who` answer multi-hop relational queries (e.g. *"designers who
led Milan houses"*, *"path from Raf Simons to Milan"*) that a vector store
**structurally cannot**.

*(Caveats: coverage is a same-corpus recall metric; the judge is a local 7B model —
a stronger judge / human panel is future work. See `fg/kg/evaluate.py`.)*

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # core + dev tools
pip install -e ".[rag,vision]"    # RAG + vision extras
pytest                            # 124 passing

# Local LLM (recommended on Apple Silicon):
brew install ollama
ollama pull qwen2.5:7b-instruct   # text (bootstrap / analyze / KG)
ollama pull qwen2.5vl:7b          # vision (look review) — or llama3.2-vision
# …or use the API: put OPENAI_API_KEY in .env and pass --backend openai
```

The command is **`fgraph`** (not `fg` — that collides with the shell builtin);
`python -m fg.cli …` works too.

## Usage

```bash
fgraph info                                          # resolved config

# Knowledge core
python scripts/download_datasets.py --sets core      # fetch data (HF, no Kaggle)
fgraph data build --source wikipedia,fashion_products,style_instruct
fgraph data smoke "quiet luxury tailoring"

# Capabilities
fgraph bootstrap --answers examples/brand_answers.example.json --out brand.md
fgraph analyze "gorpcore" --depth expert
fgraph look outfit.jpg --occasion "wedding" --out review.md   # VLM sees the photo

# Vision indexes (heavy, one-time, GPU)
fgraph vision build            # product visual index (similar pieces)
fgraph vision build-runway     # runway index → image↔image designer lineage

# Knowledge graph
fgraph kg build --limit 100                          # extract triples (uses the LLM)
fgraph kg stats
fgraph kg query "Prada"                              # facts for one entity
fgraph kg who based_in Milan                         # one-hop relational filter
fgraph kg path "Raf Simons" "Prada"                  # multi-hop reasoning
fgraph kg predict "Celine" -k 5                      # one-shot ICL link prediction
fgraph kg eval -n 8 --judge                          # KG-vs-RAG lift experiment

# The router picks the capability itself
fgraph route "help me start a quiet-luxury knitwear label"
```

Training (Colab/M4): `python -m fg.training.train_aesthetic --sources surrey`
(aesthetic scorer). Local models: `ollama pull qwen2.5vl:7b` for the vision review.

## Layout

```
fg/
├── brain/          # router, context_builder, output_contract, memory
├── capabilities/   # understand (trend), strategize (bootstrapper), personal_stylist (look review)
├── llm/            # provider-agnostic LLM interface (text + vision) + backends
├── kg/             # schema+resolution, SQLite store, extractor, reasoning, evaluate, link_prediction
├── rag/            # indexer, retriever, visual_retriever, fusion (RRF), embeddings
├── vision/         # embedder (Marqo), segmentation, index, aesthetics, movements, kg_linker, runway
├── models/         # CLIP encoder, Temporal GNN (parked trend module)
├── data/           # ingest pipeline (schema, clean, sources) + CLI
├── training/       # aesthetic scorer, CLIP fine-tune, pair sources
└── api/            # FastAPI + WebSocket (Phase 6, not yet built)
```

## Roadmap from here (multimodal-KG core — see NOVEL_IDEAS.md)

- **Grounding eval**: held-out *designer top-k accuracy* for the runway linker — a
  clean quantitative result for the thesis.
- **VLM visual extraction**: run the VLM over the runway images to generate the
  missing per-look captions *and* image-grounded KG edges.
- **Visual node prototypes**: per-designer visual centroid attached to each KG
  node (the literal multimodal KG node).
- **Cross-modal alignment**: align graph-structural embeddings (node2vec/TransE)
  with the SigLIP space — the deepest, most publishable grounding.
- **Later**: worn-together edges (Polyvore → Complete-the-Look), FashionKLIP
  embedder fine-tune, trend-forecasting (repurpose the Temporal GNN on the runway
  seasons), FastAPI + tldraw canvas.
