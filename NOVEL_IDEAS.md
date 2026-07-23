# FashionGraph — Novel Ideas & Research Directions

Living doc for the non-obvious, thesis-worthy directions. Ordered roughly by
feasibility. The north star: **a multimodal fashion knowledge graph** where the
symbolic reasoning graph and the visual world are fused, so the system can
*ground an outfit photo in the graph* and *reason with both*.

---

## The core problem

The KG is **symbolic** (entities + relations mined from text). The vision layer
is **KG-less** (embeddings, no relations). Path A currently bridges them by
matching a look image to *text descriptors* of KG nodes — indirect and lossy.
We want **direct visual grounding**: look image → graph.

Most published fashion KGs are purely symbolic; most fashion vision models have
no KG. **Joining them — nodes that carry both relational facts and a visual
representation — is the underexplored contribution.**

---

## Available data (better than expected)

`data/raw/vogue_runway/` — **~2,221 runway images, 11 houses**
(bottega-veneta, celine, gucci, balenciaga, prada, marni, jacquemus,
alexander-mcqueen, rick-owens, loewe, acne-studios), organised
`brand/season-year-type/`. Each image's JSON has `designer, show, season, type,
image_path` — **clean labels.** Missing: per-look garment captions (the VLM can
generate these). This unlocks the image-based mechanisms below *now*.

---

## Ways to achieve visual grounding (many; combine them)

### 1. Runway visual index → image↔image linking ⭐ (do first — buildable now)
Embed all 2,221 runway images with FashionSigLIP into a `VisualIndex` (reuse the
`build_product_index` pattern), metadata = designer/collection/season. Then a
look photo → **k-NN nearest real runway looks** → aggregate their designers/
collections → traverse the KG for lineage. This turns Path A from image↔*text*
into image↔*image against real designer looks* — accurate, non-parametric, no
training. This is the single highest-leverage upgrade.

### 2. Visual node prototypes (multimodal KG nodes) ⭐
For each designer/aesthetic KG node, store a **visual centroid** = mean
FashionSigLIP embedding of its runway images. Attach to the node (parallel vector
keyed by entity). Look→KG = look embedding vs node centroids. Coarser than #1 but
gives every node a visual identity → the literal "multimodal KG." Combine: #2 for
coarse designer match, #1 (k-NN) for fine collection match.

### 3. VLM-extracted visual edges (buildable now; fixes "no captions")
Run the VLM over each runway image → structured triples
(silhouette, palette, material, aesthetic, mood) → add as **image-grounded
edges** and as the missing per-look captions. The VLM does double duty: reviews
user looks *and* builds the visual KG. Bonus: caption → text embed → also
enriches RAG.

### 4. Per-collection hierarchy + temporal nodes
Model brand → collection(season) → look as graph levels; collection nodes get
their own visual centroid. Enables "which *collection* does this resemble" and,
with season/year, **trend evolution over time** (feeds the Temporal-GNN idea).

### 5. Cross-modal embedding alignment (most novel; needs training)
Learn a small projection aligning the KG's **structural** embeddings
(node2vec / TransE over the graph) with the **visual** SigLIP space, so a look
image maps into the graph's structural neighbourhood — not just nearest visual
node, but nearest *in relational structure*. Genuine cross-modal KG grounding; a
real research contribution. Training pairs come free from the runway data
(image ↔ its designer node).

### 6. Aesthetic concept axes (interpretable grounding)
Define bipolar axes (minimal↔maximal, structured↔fluid, tonal↔saturated,
classic↔avant-garde) as text anchors in SigLIP space; project both looks and
designers onto them. Gives an **interpretable coordinate** for "this look is
here; Rick Owens is over there" — explainable, and pairs with the aesthetic scorer.

### 7. FashionKLIP-style concept alignment (embedder fine-tune)
Fine-tune the embedder with concept-level alignment using the KG's concepts as
supervision (the deferred FashionKLIP option). Sharpens *every* image↔concept
match above. Do only if #1–#3 prove too loose.

---

## How we finish it now → then improve

**MVP (now, no training, uses vogue_runway):**
1. Build the **runway visual index** (#1) from `data/raw/vogue_runway`.
2. Add a **RunwayLinker**: look → k-NN runway looks → designers/collections →
   KG traversal → lineage. Wire into `LookReview` beside `KGEntityLinker`.
3. Compute **per-node visual centroids** (#2) and attach to KG entities.
4. Result: visually-grounded Path A — "this reads Rick Owens / Margiela, nearest
   to their FW-XX looks; lineage traces to …" — from *real images*, not text.

**Improve (later):**
5. **VLM visual extraction** (#3) → image-grounded edges + captions (fixes metadata).
6. **Per-collection + temporal** nodes (#4) → trend evolution.
7. **Cross-modal alignment** (#5) and/or **FashionKLIP fine-tune** (#7) → the
   deepest, most publishable grounding.

---

## Thesis framing

> A **multimodal fashion knowledge graph**: nodes carry symbolic relational facts
> *and* visual prototypes (from labeled runway imagery), with visually-extracted
> edges (VLM) and, optionally, a learned cross-modal alignment. This enables
> **image→graph grounding** (ground a look in the graph) and **graph-grounded
> visual reasoning** (explain a look via designer lineage and relations). It is
> the synthesis of the three FashionGraph layers — symbolic KG, vision embedder,
> and VLM — and targets a gap: fashion KGs are symbolic, fashion vision is
> KG-less; joining them is underexplored.

## Grounding in the MMKG literature (citations + terminology)

Our approach is an instance of **Multimodal Knowledge Graphs (MMKG)** — an
established area. Adopt its terminology and cite it; the *novelty* is MMKG applied
to **fashion reasoning + stylist grounding**, which is underexplored.

- **"Mirror node"** = attach an image embedding to a textual entity node (What-Is-
  a-MMKG survey). This is exactly our **visual node prototype** (#2) — use this term.
- **KG4MM vs MM4KG** taxonomy (*Knowledge Graphs Meet Multi-Modal Learning*,
  arXiv:2402.05391): KG-helps-vision vs vision-helps-KG. FashionGraph does **both**
  (KG grounds the stylist *and* runway images ground/extend the KG) — a clean framing.
- **Annotation-free MMKG construction** (arXiv:2503.12972) — justifies our VLM
  visual-extraction (#3): build image-grounded edges without manual labels.
- **MR-MKG** (arXiv:2406.02030) — relation-graph attention + cross-modal alignment
  module → the citation/blueprint for our cross-modal alignment (#5).
- **MMGraphRAG** (arXiv:2507.20804) / MMKG-enhanced RAG (arXiv:2508.05318) →
  our KG-grounded retrieval, extended to vision.
- Surveys / repos for Related Work: MMKG survey (arXiv:2202.05786), KG-MM-Survey
  (github zjukg/KG-MM-Survey), awesome-multimodal-knowledge-graph.

Suggested stack (from the doc) ≈ ours: CLIP/fashion-ViT (Marqo ✓), LLM extraction
(✓), graph DB with **mirror nodes** (we use SQLite + a parallel vector index, not
Neo4j), contrastive/cross-attention alignment (future #5), MMGraphRAG layer (✓).

## Deferred — fabric *visual* texture mirror nodes

**Status: parked (dataset too large for now).** The text-level fabric layer is
DONE (`fg/kg/fabrics.py` — 29 fabrics × properties/texture/season, in the KG).
The *visual* texture grounding — a mean texture embedding per fabric (mirror
node) so a swatch/garment-crop → fabric — is **built but unfed**:
`fg/vision/fabric_texture.py` (`build_texture_index`, `FabricTextureLinker` with
`identify()` + `centroids()`), `fgraph vision build-textures <dir>`, dataset-agnostic
(folder-per-fabric).

To resume later:
- **text2fabric** (valentin.deschaintre.fr/text2fabric) is the target — 3k
  materials / 45k images + text descriptions, colour, multimodal. But it's
  multi-GB and organized as (image, free-text description) pairs, NOT
  `<fabric>/*.jpg`. Needs a small **adapter** that maps each description →
  a fabric-ontology name, then feeds `build_texture_index`.
- **DTD** (folder-per-class ZIP) is the zero-friction smoke-test (generic
  textures, not named fabrics) — works with `vision build-textures` as-is.
- Rejected: TILDA (greyscale, defect-detection, 1996 — wrong domain).

Honest caveat when resumed: fabric ID needs close-up crops (segmenter garment
crops), not full-outfit photos.

## Open decisions
- Prototype (centroid) vs k-NN vs both for look→node — start with **both**
  (centroid coarse, k-NN fine), measure.
- Licensing: the Vogue images are for **research/thesis** use; do not redistribute.
- Evaluate grounding: hold out some runway looks, check the linker recovers the
  right designer (a real, reportable metric — "designer top-k accuracy").
