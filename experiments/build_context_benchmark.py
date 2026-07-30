"""
Experiment 3 (rung 1) — context-conditional minimal-pair benchmark
==================================================================

Taste is not a beauty scalar -- it is "right FOR this context." Hold the IMAGE
perfectly fixed and change only the CONTEXT. A context-conditional scorer must
change its verdict; a per-image beauty/confound scalar CANNOT (same image ->
same score), so it scores exactly 0.5 by construction. No image editing, so no
generative artifacts.

A record is a minimal pair (image, context_good, context_bad). Ground truth
never comes from the model under test:

  1. RUNWAY captions -- full looks; garment cue -> season/formality rules.
  2. FASHION-PRODUCT-IMAGES (44k, parquet) -- single garments with EXPLICIT
     labels `usage` (Formal/Sports/Ethnic) and weather-signalling `articleType`.

Contexts are SAMPLED from rich pools of realistic occasions (deterministically
per image id, so it is reproducible but varied) and phrased with several
templates -- no endlessly repeated "beach / wedding" strings. The good context
is drawn from a pool the garment genuinely suits; the bad context from a pool it
clearly does not.

Product images are referenced as ``fpi:{id}`` and decoded from the parquet at
eval time (see fpi_loader.py) -- no thousands of files extracted.

    python experiments/build_context_benchmark.py \
        --out experiments/out/context_benchmark.jsonl
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

# ---------- rich occasion pools ------------------------------------------- #
POOLS = {
    "warm": [                                   # hot-weather occasions
        "a hot day at the beach", "a tropical summer holiday",
        "a poolside afternoon", "an outdoor summer festival",
        "a sweltering city afternoon", "a summer garden party",
        "a seaside vacation", "a warm spring picnic",
    ],
    "cold": [                                   # cold-weather occasions
        "a snowy winter morning", "a freezing winter commute",
        "a ski trip in the mountains", "a cold December evening",
        "a frosty walk outdoors", "a chilly autumn hike",
        "a bitter cold night out", "a winter holiday in the north",
    ],
    "formal": [                                 # dressed-up occasions
        "a black-tie gala", "a formal wedding ceremony",
        "a corporate boardroom meeting", "a job interview",
        "an awards dinner", "an evening at the opera",
        "a diplomatic reception", "a business conference",
    ],
    "athletic": [                               # sporty / very casual
        "an intense gym workout", "a morning run in the park",
        "a yoga class", "a game of tennis", "a weekend trail hike",
        "a basketball match", "a cycling session", "a beach volleyball game",
    ],
    "traditional": [                            # cultural / ceremonial
        "a traditional wedding celebration", "a cultural festival",
        "a religious ceremony", "a Diwali celebration",
        "a family festival gathering", "a temple visit",
    ],
}

TEMPLATES = [
    "an outfit for {}",
    "what you would wear to {}",
    "dressed for {}",
    "an outfit suited to {}",
    "clothing appropriate for {}",
]


def _h(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest(), 16)


def ctx(pool, key, role):
    """Deterministically sample an occasion + template -> a full prompt."""
    occ = POOLS[pool][_h(key, pool, role) % len(POOLS[pool])]
    tmpl = TEMPLATES[_h(key, role, "t") % len(TEMPLATES)]
    return tmpl.format(occ)


def pair(records, axis, source, ref, key, good_pool, bad_pool, cue):
    records.append(dict(
        axis=axis, source=source, image=ref, cue=cue,
        context_good=ctx(good_pool, key, "good"),
        context_bad=ctx(bad_pool, key, "bad"),
    ))


# ---------- source 1: runway captions ------------------------------------- #
COLD_CUES = ["wool coat", "peacoat", "puffer", "parka", "trench", "overcoat",
             "heavy knit", "turtleneck", "shearling", "wool", "cashmere", "coat"]
WARM_CUES = ["swimwear", "bikini", "linen", "tank top", "shorts", "sundress",
             "sleeveless"]
FORMAL_CUES = ["suit", "tailored", "tuxedo", "gown", "blazer", "velvet",
               "silk", "evening", "pinstriped", "sharp"]
CASUAL_CUES = ["denim", "hoodie", "sweatpants", "sneakers", "graphic",
               "bomber", "sport", "track"]


def _cue(text, cues):
    for c in cues:
        if c in text:
            return c
    return None


def from_runway(path, out):
    if not os.path.exists(path):
        return 0
    n = 0
    for i, line in enumerate(open(path)):
        c = json.loads(line)
        t = c["caption"].lower()
        img = c.get("image_path", "")
        key = img or f"rw{i}"
        cold, warm = _cue(t, COLD_CUES), _cue(t, WARM_CUES)
        formal, casual = _cue(t, FORMAL_CUES), _cue(t, CASUAL_CUES)
        if cold and not warm:
            pair(out, "season", "runway", img, key, "cold", "warm", cold); n += 1
        elif warm and not cold:
            pair(out, "season", "runway", img, key, "warm", "cold", warm); n += 1
        if formal and not casual:
            pair(out, "formality", "runway", img, key, "formal", "athletic", formal); n += 1
        elif casual and not formal:
            pair(out, "formality", "runway", img, key, "athletic", "formal", casual); n += 1
    return n


# ---------- source 2: fashion-product-images ------------------------------ #
COLD_ARTICLES = {"Sweaters", "Sweatshirts", "Jackets", "Coats",
                 "Nehru Jackets", "Rain Jacket"}
WARM_ARTICLES = {"Shorts", "Sandals", "Flip Flops", "Swimwear", "Capris",
                 "Sarongs"}


def from_products(pattern, out):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(pattern))
    if not files:
        return 0
    n = 0
    cols = ["id", "usage", "articleType", "masterCategory", "productDisplayName"]
    for f in files:
        for r in pq.read_table(f, columns=cols).to_pylist():
            if r["masterCategory"] not in ("Apparel", "Footwear"):
                continue
            ref, key = f"fpi:{r['id']}", r["id"]
            atype, usage = r["articleType"], r["usage"]

            if usage == "Formal":
                pair(out, "formality", "fpi", ref, key, "formal", "athletic",
                     f"usage=Formal/{atype}"); n += 1
            elif usage == "Sports":
                pair(out, "formality", "fpi", ref, key, "athletic", "formal",
                     f"usage=Sports/{atype}"); n += 1
            elif usage == "Ethnic":
                pair(out, "occasion", "fpi", ref, key, "traditional", "athletic",
                     f"usage=Ethnic/{atype}"); n += 1

            if atype in COLD_ARTICLES:
                pair(out, "season", "fpi", ref, key, "cold", "warm",
                     f"article={atype}"); n += 1
            elif atype in WARM_ARTICLES:
                pair(out, "season", "fpi", ref, key, "warm", "cold",
                     f"article={atype}"); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/processed/runway_captions.jsonl")
    ap.add_argument("--products",
                    default="data/raw/fashion-product-images-small/data/*.parquet")
    ap.add_argument("--out", default="experiments/out/context_benchmark.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    records = []
    n_rw = from_runway(args.captions, records)
    n_fpi = from_products(args.products, records)

    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    from collections import Counter
    by_axis = Counter(r["axis"] for r in records)
    uniq_ctx = len({r["context_good"] for r in records}
                   | {r["context_bad"] for r in records})
    print(f"runway {n_rw}  product {n_fpi}  TOTAL {len(records)} pairs")
    print(f"  by axis: {dict(by_axis)}")
    print(f"  distinct context phrases: {uniq_ctx}")
    print(f"wrote {args.out}")
    print("\nsamples:")
    for r in records[105:110] + records[-3:]:
        print(f"  [{r['axis']:9s}] good='{r['context_good']}'  |  "
              f"bad='{r['context_bad']}'")


if __name__ == "__main__":
    main()
