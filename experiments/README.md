# Taste experiments

Isolated research branch (`experiments/taste-confound-baseline`). Nothing here
touches the app — it is the scientific spine of the "does the machine have
taste, or just good lighting?" study that came out of the LLM council.

## Why this exists

The council's load-bearing worry (the Contrarian / Outsider advisors) was
**construct validity**: a fashion "taste" scorer trained on photo-preference
data may just be a *production-quality detector* — it rewards brightness,
sharpness, resolution, brand cues — and calls it taste. Before building any
taste model we have to know how much of the human preference signal is
explained by dumb low-level confounds alone. That floor is what every richer
model must beat to make a falsifiable claim.

## Experiment 1 — confound-only baseline

`confound_baseline.py` fits a linear Bradley-Terry model
(`P(A beats B) = sigmoid(u(A) − u(B))`, `u(x) = w·features(x)`) on the
**Surrey "Aesthetics Based on Fashion Images"** pairwise data (70,000
"which look is better" judgments, 1,064 images, 120 body-shape × top × bottom
configs, 10 fashion-follower annotators + 1 expert).

The features are **only low-level confounds** — no embeddings, no semantics,
no KG: brightness, contrast, colourfulness, saturation, sharpness
(variance-of-Laplacian), edge density, luminance entropy, log-resolution,
aspect ratio, colour warmth. Pure numpy + Pillow so it runs anywhere.

Two honesty controls:

* **Image-disjoint split** — images in the test comparisons are never seen in
  training (also a stricter config-disjoint split is reported).
* **Human ceiling** — the *honest* ceiling is inter-annotator agreement on the
  1,400 pairs judged in more than one set. (The naive "predict per-pair
  majority" number, 0.946, is inflated: most pairs are judged once, so it
  scores them 100% by construction. We ignore it.)

### Result

| quantity | accuracy |
|---|---|
| chance / majority class | 0.537 |
| **confound-only model (image-disjoint)** | **0.573** |
| confound-only model (config-disjoint) | 0.625 |
| **human agreement (honest ceiling)** | **0.732** |
| naive per-pair-majority ceiling *(inflated, ignore)* | 0.946 |

Strongest single confound is luminance **entropy** (0.578 univariate); the full
model's largest weights are brightness (+0.23), aspect (+0.19), saturation
(+0.14) — brighter, taller-cropped, more saturated images win *slightly*.

### Reading

On Surrey, **low-level photo quality barely beats chance (+0.036) and sits far
below human agreement (0.732).** Confounds do *not* explain the preference —
roughly **0.16 accuracy points** of human-agreed signal live above the confound
floor and are, by elimination, *semantic*. For this dataset the Contrarian's
worst case does **not** hold: there is real, non-photographic taste signal to
model. That is a genuinely encouraging (and falsifiable) finding.

### The caveat that keeps it honest

Surrey is a **controlled** set — standardised body-shape renderings that
*deliberately* lack the brand logos, luxury cues and professional runway
photography that contaminate real runway imagery. So a low confound floor here
does **not** clear the ~2,221-image **runway** set. The runway set has no
pairwise labels and is exactly where confounds bite hardest, so it needs its
own **confound-controlled minimal-pair** evaluation (same look, change one
aesthetic variable, hold photo/brand/quality fixed) — the benchmark every peer
reviewer independently said the field is missing. That is Experiment 3.

## What's next

**Experiment 2 — frozen-feature critic (the payoff test).**
Fit the *same* Bradley-Terry head on **frozen FashionSigLIP** embeddings of the
Surrey images, on the *same image-disjoint split*. If it climbs from 0.573
toward 0.732, we have direct evidence that frozen fashion features carry a
taste signal the confounds don't — the money finding. (Needs Surrey embeddings;
the runway embeddings already exist at `data/embeddings/runway_fashionsiglip.npz`.)

**Experiment 3 — confound-controlled minimal-pair benchmark (the flagship).**
Generate controlled edits (swap shoes / break colour logic / wrong occasion)
holding lighting, model, resolution and brand fixed; collect expert-rubric or
human labels; test whether any scorer tracks the *edit* while a confound-only
model cannot. Publishable either way.

## Run

```bash
python experiments/confound_baseline.py \
    --surrey data/raw/surrey-aesthetics \
    --out    experiments/out/confound_baseline
```

Outputs `experiments/out/confound_baseline_results.json` and a cached
`*_features.npz` (delete to recompute features).
