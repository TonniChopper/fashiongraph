"""
Experiment 2.5 — what IS the taste axis?  (concept-axis probe)
=============================================================

Experiment 2 showed a frozen-feature critic reaches 0.684 (vs 0.573 confound
floor, 0.732 human) and that the signal is low-rank. But "a taste signal
exists" is not yet interpretable. This asks: WHAT is that signal made of?

Method (no labelling, no VLM, no training of the encoder):
FashionSigLIP is a shared image/text space, so a named style concept is just a
DIRECTION defined by two text prompts. For each concept we take
    axis = normalise( enc("a {positive} outfit") - enc("a {negative} outfit") )
and project every (frozen) image embedding onto it -> an interpretable
coordinate. Then:

  (a) INTERPRETABLE TASTE MODEL: fit the same Bradley-Terry head on the handful
      of concept coordinates (image-disjoint split). How close does a dozen
      *named* axes get to the opaque 0.684 full-frozen critic? The gap is how
      much of taste these concepts explain vs. what stays unnamed.
  (b) WHAT WINS: per-concept univariate accuracy + signed weight -> e.g.
      "more polished / more harmonious colour / more tailored looks win."

This turns "the machine has taste" into "here is what its taste is made of" --
the interpretability a thesis needs.

Prereq: data/embeddings/surrey_fashionsiglip.npz (from embed_surrey.py).
Text encoding needs the FashionSigLIP model -> run on the M4.

    python experiments/concept_probe.py \
        --emb data/embeddings/surrey_fashionsiglip.npz \
        --out experiments/out/concept_probe
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from confound_baseline import (
    load_comparisons, fit_logistic, accuracy, design,
)

CONFOUND_FLOOR = 0.573
FROZEN_FULL = 0.684
HUMAN_CEILING = 0.732

# name -> (positive prompt, negative prompt).  Interpretable style axes.
CONCEPTS = [
    ("minimal_vs_ornate",      "a minimalist, clean, understated outfit",
                               "a busy, ornate, over-decorated outfit"),
    ("tailored_vs_shapeless",  "a sharply tailored, structured outfit",
                               "a shapeless, ill-fitting outfit"),
    ("colour_harmony",         "an outfit with harmonious, coordinated colours",
                               "an outfit with clashing, mismatched colours"),
    ("polished_vs_sloppy",     "a polished, well-put-together outfit",
                               "a sloppy, unfinished-looking outfit"),
    ("proportion",             "an outfit with balanced, flattering proportions",
                               "an outfit with awkward, unbalanced proportions"),
    ("elegant_vs_tacky",       "an elegant, refined, sophisticated outfit",
                               "a tacky, garish, cheap-looking outfit"),
    ("classic_vs_dated",       "a timeless, classic outfit",
                               "a dated, unfashionable outfit"),
    ("cohesive_vs_random",     "a cohesive, intentional outfit",
                               "a random, thrown-together outfit"),
    ("modern_vs_frumpy",       "a modern, current outfit",
                               "a frumpy, outdated outfit"),
    ("luxurious_vs_cheap",     "a luxurious, high-quality outfit",
                               "a cheap, low-quality outfit"),
]


def concept_axes(names):
    """Embed prompt pairs with FashionSigLIP text encoder -> unit axes (K, D)."""
    from fg.vision.embedder import FashionEmbedder
    emb = FashionEmbedder()
    pos = emb.encode_texts([p for _, p, _ in CONCEPTS]).astype(np.float64)
    neg = emb.encode_texts([n for _, _, n in CONCEPTS]).astype(np.float64)
    ax = pos - neg
    ax /= (np.linalg.norm(ax, axis=1, keepdims=True) + 1e-8)
    return ax  # (K, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/embeddings/surrey_fashionsiglip.npz")
    ap.add_argument("--out", default="experiments/out/concept_probe")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows, _ = load_comparisons("data/raw/surrey-aesthetics")
    d = np.load(args.emb, allow_pickle=True)
    names_img = list(d["names"])
    E = d["embeddings"].astype(np.float64)
    img_emb = {n: E[i] for i, n in enumerate(names_img)}

    axes = concept_axes([c[0] for c in CONCEPTS])          # (K, D)
    cnames = [c[0] for c in CONCEPTS]

    # interpretable coordinates: project each image onto every concept axis
    feat = {n: axes @ v for n, v in img_emb.items()}       # (K,) per image

    # same image-disjoint split as Exp 1/2
    rng = np.random.default_rng(args.seed)
    all_imgs = np.array(sorted(img_emb))
    rng.shuffle(all_imgs)
    nt = int(len(all_imgs) * args.test_frac)
    test_imgs = set(all_imgs[:nt].tolist())
    train_imgs = set(all_imgs[nt:].tolist())
    train_rows = [r for r in rows if r[0] in train_imgs and r[1] in train_imgs]
    test_rows = [r for r in rows if r[0] in test_imgs and r[1] in test_imgs]

    Ftr = np.stack([feat[i] for i in train_imgs])
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Xtr, ytr = design(train_rows, feat, mu, sd)
    Xte, yte = design(test_rows, feat, mu, sd)
    w = fit_logistic(Xtr, ytr, l2=1e-2, epochs=500, seed=args.seed)
    acc_concept = accuracy(Xte, yte, w)

    # per-concept univariate accuracy + signed contribution
    uni = {}
    for j, name in enumerate(cnames):
        wj = np.zeros(len(cnames)); wj[j] = 1.0
        a = accuracy(Xte, yte, wj)
        # orient so "positive concept present" is the +direction
        uni[name] = {"univariate_acc": max(a, 1 - a),
                     "wins_when": "more" if a >= 0.5 else "less",
                     "fitted_weight": float(w[j])}

    ranked = sorted(uni.items(), key=lambda kv: -kv[1]["univariate_acc"])

    print("\n" + "=" * 66)
    print("CONCEPT-AXIS PROBE  (image-disjoint)")
    print(f"  confound floor           : {CONFOUND_FLOOR:.3f}")
    print(f"  {len(cnames)} named concepts (BT head) : {acc_concept:.3f}")
    print(f"  full frozen (Exp 2)      : {FROZEN_FULL:.3f}")
    print(f"  human agreement          : {HUMAN_CEILING:.3f}")
    explained = (acc_concept - CONFOUND_FLOOR) / max(1e-9, FROZEN_FULL - CONFOUND_FLOOR)
    print(f"  -> {len(cnames)} words explain {explained*100:.0f}% of what the "
          f"opaque critic found above confounds")
    print("\n  what wins (ranked by univariate accuracy):")
    for name, s in ranked:
        print(f"    {name:20s} {s['univariate_acc']:.3f}  "
              f"({s['wins_when']} -> better, w={s['fitted_weight']:+.2f})")
    print("=" * 66)

    json.dump({
        "confound_floor": CONFOUND_FLOOR,
        "frozen_full": FROZEN_FULL,
        "human_ceiling": HUMAN_CEILING,
        "concept_head_accuracy": acc_concept,
        "fraction_of_frozen_explained": explained,
        "per_concept": uni,
        "concepts": {c[0]: {"pos": c[1], "neg": c[2]} for c in CONCEPTS},
    }, open(args.out + "_results.json", "w"), indent=2)
    print(f"wrote {args.out}_results.json")


if __name__ == "__main__":
    main()
