# FashionGraph — Dataset Sourcing (Phase 1)

Evaluated three listing sources and picked the datasets worth the disk space,
mapped to what each FashionGraph layer actually needs. Principle from the
rebuild plan: **quality-first, free, and directly usable** — not bulk.

## Verdict on the three sources

| Source | Verdict |
|---|---|
| **data4fashion (Medium)** | ✅ Useful. ~25 free datasets (Kaggle/data.world). The practical backbone of our sourcing. |
| **Surrey "Aesthetics Based on Fashion Images"** | ✅ Niche but unique. Small (1,064 imgs) but has body-shape + human aesthetic preference labels found nowhere else. |
| **Datarade** | ❌ Skip for now. Commercial marketplace — all paid vendor feeds (PromptCloud, 42Signals, Gener8/Temu e-receipts). Revisit only for the B2B enterprise phase *with budget*. |

---

## The picks (ranked)

### 1. Fashion Product Images Dataset — *the backbone* ⭐
Kaggle: `paramaggarwal/fashion-product-images-dataset` (full ~25 GB) and
`…-small` (~280 MB, low-res).
- **~44k images** + `styles.csv` with `gender, masterCategory, subCategory,
  articleType, baseColour, season, year, usage, productDisplayName`.
- **Why it wins:** one dataset feeds three layers — image-text pairs for CLIP
  fine-tune, clean structured attributes for the KG / DNA encoder, and
  `season`+`year` for a light temporal signal. Start with `-small` on the M4.
- Feeds: KG core, CLIP (Phase 5), attribute extraction.

### 2. H&M Personalized Fashion Recommendations — *the temporal signal* ⭐
Kaggle competition: `h-and-m-personalized-fashion-recommendations` (~35 GB).
- **~105k product images** + `articles.csv` (rich metadata) + **`transactions`
  with dates** + customers.
- **Why:** the only pick here with real **time-stamped purchase history** —
  the right fuel for the Temporal GNN (Phase 4) and demand/trend estimation.
- Note: requires accepting Kaggle competition rules. Large — pull metadata +
  a sampled image subset first.
- Feeds: Temporal GNN, trend/demand, co-occurrence KG edges.

### 3. Surrey "Aesthetics Based on Fashion Images" — *Personal Stylist secret weapon* ⭐
Direct zip: `kahlan.eps.surrey.ac.uk/featurespace/fashion/fashion_data.zip`
- 1,064 images across 120 configs = **body shape** (apple/column/hourglass/pear)
  × **top** × **bottom** clothing, plus **~70k pairwise aesthetic judgments**
  from 10 fashion-following annotators + an expert.
- **Why:** body-shape + subjective "which looks better" labels are exactly what
  Personal Stylist look-review needs and nothing else here provides. 
- Caveats: small, dated (2014 styling). Use as a **specialized fine-tune / eval
  set for an aesthetic scorer**, not as backbone imagery.
- Feeds: Personal Stylist (occasion/silhouette/body fit), aesthetic scoring.

### 4. Women's E-Commerce Clothing Reviews — *language & cultural context*
Kaggle: `nicapotato/womens-ecommerce-clothing-reviews` (~23k rows).
- Reviews + rating, recommended flag, division/department/class.
- **Why:** clean, classic text dataset → grounds the Fashion LLM / RAG in real
  customer language and sentiment; good for cultural-context retrieval.
- Feeds: RAG corpus, Fashion LLM tone, sentiment.

### 5. Myntra Fashion Product Dataset — *catalog + brand breadth*
Kaggle: `hiteshsuthar101/myntra-fashion-product-dataset` (also `shivamb/…`).
- Large catalog: price, name, colour, brand, rating.
- **Why:** widens attribute + **brand-node** coverage beyond Western/adidas-nike
  bias; complements #1 for the KG.
- Feeds: KG breadth, brand DNA.

### 6. Clothing Co-Parsing / Clothes Segmentation — *per-item segmentation (optional)*
Kaggle: `balraj98/clothing-coparsing-dataset`.
- Pixel-level garment masks.
- **Why:** supports the "segment before embedding" approach (segformer) so one
  outfit photo → per-item nodes. We use pretrained `segformer_b3_clothes`, so
  this is **eval/fine-tune only** — nice-to-have.
- Feeds: Personal Stylist segmentation, Look Analysis.

### 7. Social / trend text — *signal layer (nice-to-have, noisy)*
- Nike **#JustDoIt tweets** (`eliasdabbas/5000-justdoit-tweets-dataset`).
- **Instagram fashion conversations** (Harvard Dataverse `K7AW6F`).
- **Why:** raw trend/virality signal. Noisy — defer until the core is solid.
- Feeds: Viral Potential Score, trend signal.

---

## Skip / deprioritize
- **All Datarade feeds** — paid; enterprise phase only.
- **Fashion-MNIST** — 28×28 grayscale toy; sanity-check only, not real fashion.
- **Single-brand / tiny sets** (Gymboree 395 rows, Adidas 1.5k, Clothes-Size-
  Prediction) — too narrow for a general KG.
- Heavyweight academic sets **DeepFashion / DeepFashion2 / Polyvore** (not in
  these three lists but in the brief) remain the go-to for landmarks / attribute
  richness / outfit compatibility — pull later if #1–#2 fall short.

## Suggested Phase-1 download order (M4-friendly)
1. **Fashion Product Images – small** → build the first KG + attribute pipeline end-to-end cheaply.
2. **Women's E-Commerce Reviews** → seed the RAG text corpus.
3. **H&M** (metadata + sampled images) → temporal + co-occurrence edges.
4. **Surrey Aesthetics** → Personal Stylist aesthetic scorer.
5. Myntra / segmentation / social → breadth and specialized signals as needed.

---

# Textual data for LLM training / understanding

Goal: make the Fashion LLM *speak fashion* — vocabulary, styling logic, brand/era
knowledge — via (a) instruction fine-tune data and (b) a domain corpus for RAG +
optional continued pretraining.

**Reality check:** large high-quality *text-only* fashion instruction sets barely
exist. Best practice (per UniFashion, Interactive-Fashion papers): seed from the
few real ones, then **generate the rest** from our structured catalog + KG via an
LLM (self-instruct / distillation). The old repo already started this
(`generate_qa_pairs.py` → expert_pairs.jsonl) — keep that pattern, done properly.

### A. Ready-made instruction / dialogue data (seeds)
- **neuralwork/fashion-style-instruct** ⭐ — 3.2k triples `input` (body type +
  personal style) / `context` (event) / `completion` (GPT-3.5 outfit rec). MIT,
  2.6 MB, text-only. *The* best pure-text seed for Personal Stylist + Brand
  Bootstrapper tone. → `hf.co/datasets/neuralwork/fashion-style-instruct`
- **lihicarmeli/fashion-stylist-multimodal-v2** — stylist triples (multimodal);
  mine the text side for styling rationales.
- **innople/fashion_design_qa** — design Q&A (imagefolder; text extractable).

### B. Product-description text (vocabulary + captioning language)
- **Marqo/fashion200k** ⭐ — 200k image+**text** product descriptions; rich
  attribute phrasing. Great source for the canonical vocabulary and for teaching
  garment-description language. Apache-2.0. → `hf.co/datasets/Marqo/fashion200k`
- **tomytjandra/h-and-m-fashion-caption** (+ `-12k`) — H&M garment captions;
  clean "describe this piece" language.

### C. Real human language (reviews / social) — tone + culture
- **Women's E-Commerce Clothing Reviews** (Kaggle, ~23k) — customer voice, fit
  and sentiment language.
- Nike **#JustDoIt tweets** / **Instagram fashion conversations** (Harvard
  Dataverse) — slang, trend chatter (noisy; defer).

### D. Encyclopedic knowledge (RAG core) — brands, houses, eras, fabrics
- **Wikipedia fashion corpus** ⭐ — houses, designers, decades, garments,
  fabrics, fashion weeks. This is genuinely good domain knowledge (the old
  repo's Wikipedia scrape is worth keeping even if the *runway* scrapes weren't).
  Build via `wikipedia-api` over a curated page list → chunk into the KG/RAG.

### Recommended LLM-data strategy (phased)
1. **RAG first (no fine-tune):** index corpus D + B + C into ChromaDB so the LLM
   is *grounded* before it's *tuned*. Cheap, immediate quality.
2. **Seed instructions:** start from `fashion-style-instruct` (A).
3. **Self-instruct expansion:** generate 5–20k instruction pairs from our Fashion
   Product Images attributes + Fashion200k descriptions + KG relations (tasks:
   attribute→description, look critique, occasion styling, brand-DNA Q&A). Filter
   with a quality/consistency pass. This becomes the LoRA training set (Phase 5).
4. **LoRA fine-tune** (MLX on the M4 / Colab) on the cleaned mix; keep RAG at
   inference so facts stay grounded and current.

---

## Licensing note
Kaggle sets are free but each has its own license/terms (H&M requires competition
rules acceptance); Surrey is academic (cite Gaur & Mikolajczyk, ICPR 2014).
Verify per-dataset terms before any redistribution or commercial use.
