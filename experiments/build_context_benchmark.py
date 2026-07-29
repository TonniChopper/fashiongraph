"""
Experiment 3 (rung 1) — context-conditional minimal-pair benchmark
==================================================================

The council's flagship idea, in its cheapest artifact-free form.

Taste is not a beauty scalar -- it is "right FOR this context." The sharpest
way to show that: hold the IMAGE perfectly fixed and change only the CONTEXT.
A context-conditional scorer must change its verdict; a per-image beauty/
confound scalar CANNOT (same image -> same score), so it scores exactly 0.5
by construction. No image editing, so no generative artifacts -- the objection
that killed diffusion-based tests does not apply.

A benchmark record is a minimal pair:
    (image, context_good, context_bad)
where the garment clearly suits context_good and not context_bad. Ground truth
comes from a TRANSPARENT rules table over garment cues in the runway caption
(this is the KG's stylist knowledge, written out and auditable) -- not from the
model we will test. Two axes with reliable vocabulary coverage:

  * SEASON     warm-layers (wool coat, puffer, knit) -> cold day, not hot beach
  * FORMALITY  tailoring (suit, gown, tuxedo)        -> formal event, not gym

This script only PARSES captions -> it needs no model and produces
benchmark.jsonl now. Scoring is done by eval_context.py on the M4.

    python experiments/build_context_benchmark.py \
        --captions data/processed/runway_captions.jsonl \
        --out      experiments/out/context_benchmark.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

# garment cue -> axis membership. Auditable "stylist rules" (stand-in for KG).
COLD_CUES = ["wool coat", "peacoat", "puffer", "parka", "trench", "overcoat",
             "heavy knit", "turtleneck", "shearling", "wool", "cashmere", "coat"]
WARM_CUES = ["swimwear", "bikini", "linen", "tank top", "shorts", "sundress",
             "sleeveless"]
FORMAL_CUES = ["suit", "tailored", "tuxedo", "gown", "blazer", "velvet",
               "silk", "evening", "pinstriped", "sharp"]
CASUAL_CUES = ["denim", "hoodie", "sweatpants", "sneakers", "graphic",
               "bomber", "sport", "track"]

SEASON = {
    "good": "a cold winter day", "bad": "a hot summer beach day",
    "flip_good": "a hot summer beach day", "flip_bad": "a cold winter day",
}
FORMAL = {
    "good": "a formal evening event", "bad": "a gym workout",
    "flip_good": "a gym workout", "flip_bad": "a formal evening event",
}


def any_cue(text, cues):
    return [c for c in cues if c in text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/processed/runway_captions.jsonl")
    ap.add_argument("--out", default="experiments/out/context_benchmark.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    caps = [json.loads(l) for l in open(args.captions)]
    records = []
    for c in caps:
        t = c["caption"].lower()
        img = c.get("image_path", "")

        cold, warm = any_cue(t, COLD_CUES), any_cue(t, WARM_CUES)
        formal, casual = any_cue(t, FORMAL_CUES), any_cue(t, CASUAL_CUES)

        # SEASON axis: assign only when unambiguous (one side fires, not both)
        if cold and not warm:
            d = SEASON
            records.append(dict(axis="season", image=img, caption=c["caption"],
                                cue=cold[0], context_good=d["good"],
                                context_bad=d["bad"], designer=c.get("designer")))
        elif warm and not cold:
            d = SEASON
            records.append(dict(axis="season", image=img, caption=c["caption"],
                                cue=warm[0], context_good=d["flip_good"],
                                context_bad=d["flip_bad"], designer=c.get("designer")))

        # FORMALITY axis
        if formal and not casual:
            d = FORMAL
            records.append(dict(axis="formality", image=img, caption=c["caption"],
                                cue=formal[0], context_good=d["good"],
                                context_bad=d["bad"], designer=c.get("designer")))
        elif casual and not formal:
            d = FORMAL
            records.append(dict(axis="formality", image=img, caption=c["caption"],
                                cue=casual[0], context_good=d["flip_good"],
                                context_bad=d["flip_bad"], designer=c.get("designer")))

    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    n_season = sum(r["axis"] == "season" for r in records)
    n_formal = sum(r["axis"] == "formality" for r in records)
    print(f"{len(caps)} captions -> {len(records)} minimal pairs "
          f"(season {n_season}, formality {n_formal})")
    print(f"wrote {args.out}")
    print("\nsample records:")
    for r in records[:4]:
        print(f"  [{r['axis']}] cue='{r['cue']}'  good='{r['context_good']}'  "
              f"bad='{r['context_bad']}'")
    print("\nNote: a per-image beauty/confound scalar scores BOTH contexts of a")
    print("pair identically -> 0.500 by construction. Any accuracy above 0.5 is")
    print("context sensitivity a scalar model cannot have. Score with eval_context.py.")


if __name__ == "__main__":
    main()
