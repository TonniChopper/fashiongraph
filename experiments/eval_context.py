"""
Experiment 3 (rung 1) — score the context-conditional benchmark
===============================================================

Reads context_benchmark.jsonl and, for each minimal pair (image,
context_good, context_bad), scores image-context compatibility with the frozen
FashionSigLIP shared space:
    s(context) = cos( enc_image(image), enc_text("an outfit for {context}") )
Accuracy = fraction of pairs with s(good) > s(bad).

The punchline is the control: a per-image beauty/confound scalar assigns ONE
number to the image regardless of context, so s(good) == s(bad) -> 0.500 exactly.
Any lift above 0.5 here is context sensitivity that a scalar taste model is
mathematically incapable of. That is the demonstration that taste != a beauty
score.

Needs the FashionSigLIP model -> run on the M4.

    python experiments/eval_context.py \
        --benchmark experiments/out/context_benchmark.jsonl \
        --out       experiments/out/context_eval
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="experiments/out/context_benchmark.jsonl")
    ap.add_argument("--out", default="experiments/out/context_eval")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    recs = [json.loads(l) for l in open(args.benchmark)]
    recs = [r for r in recs if os.path.exists(r["image"])]
    print(f"{len(recs)} scorable pairs")

    from fg.vision.embedder import FashionEmbedder
    emb = FashionEmbedder()

    # cache text embeddings for the small set of contexts
    contexts = sorted({r["context_good"] for r in recs}
                      | {r["context_bad"] for r in recs})
    tvec = {c: emb.encode_texts([f"an outfit for {c}"])[0] for c in contexts}

    per_axis = defaultdict(lambda: [0, 0])   # axis -> [correct, total]
    margins = []
    for r in recs:
        iv = emb.encode_images([Image.open(r["image"])])[0]
        sg = float(iv @ tvec[r["context_good"]])
        sb = float(iv @ tvec[r["context_bad"]])
        ok = sg > sb
        per_axis[r["axis"]][0] += int(ok)
        per_axis[r["axis"]][1] += 1
        margins.append(sg - sb)

    tot_ok = sum(v[0] for v in per_axis.values())
    tot = sum(v[1] for v in per_axis.values())

    print("\n" + "=" * 60)
    print("CONTEXT-CONDITIONAL EVAL")
    print(f"  scalar-model control (any per-image score): 0.500  [by construction]")
    for ax, (ok, n) in sorted(per_axis.items()):
        print(f"  FashionSigLIP  {ax:10s}: {ok/n:.3f}  ({ok}/{n})")
    print(f"  FashionSigLIP  {'overall':10s}: {tot_ok/tot:.3f}  ({tot_ok}/{tot})")
    print(f"  mean margin s(good)-s(bad): {np.mean(margins):+.4f}")
    print("=" * 60)
    print("Any value above 0.500 is context sensitivity a beauty scalar cannot")
    print("have -> evidence that taste is context-conditional, not a single score.")

    json.dump({
        "scalar_control": 0.5,
        "overall_accuracy": tot_ok / tot,
        "per_axis": {a: {"acc": v[0]/v[1], "n": v[1]} for a, v in per_axis.items()},
        "mean_margin": float(np.mean(margins)),
        "n_pairs": tot,
    }, open(args.out + "_results.json", "w"), indent=2)
    print(f"wrote {args.out}_results.json")


if __name__ == "__main__":
    main()
