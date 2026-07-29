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

## Experiment 2 — frozen-feature critic (the payoff test) — READY TO RUN

`frozen_critic.py` fits the **same** Bradley-Terry head on **frozen
FashionSigLIP** embeddings of the Surrey images, on the **same image-disjoint
split and seed** as Experiment 1. The only thing that changes is the input:
768-d frozen embeddings instead of the 10 confounds. If accuracy climbs from
the 0.573 confound floor toward the 0.732 human ceiling, that is direct
evidence frozen fashion features carry a taste signal the confounds do not —
the money finding. It sweeps PCA dims `{none,64,128,256}` so the number isn't
a cherry-picked projection, and reports the fraction of the confound→human gap
it closes.

Because the encoder stays **frozen** (we read only a linear head off it), this
does not re-open the Track-B negative (fine-tuning the encoder distorts its
features). Verified: the pipeline recovers a planted linear signal on synthetic
data, and scores random embeddings at chance against the real labels.

**Run it (embedding step on your M4, ~1 min; the model + MPS are already set up
from building the runway index):**

```bash
# 1. embed the 1,064 Surrey images with the project's FashionSigLIP
python experiments/embed_surrey.py \
    --surrey data/raw/surrey-aesthetics \
    --out    data/embeddings/surrey_fashionsiglip.npz

# 2. fit the frozen-feature critic and read it against the floor + ceiling
python experiments/frozen_critic.py \
    --emb    data/embeddings/surrey_fashionsiglip.npz \
    --out    experiments/out/frozen_critic
```

(The embedding step needs the FashionSigLIP weights + torch, which are
impractical to pull into the throttled sandbox — hence it runs on your Mac.)

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
