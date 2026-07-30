"""
Experiment 3 (rung 1) — context-conditional minimal-pair benchmark
==================================================================

The council's flagship idea, in its cheapest artifact-free form.

Taste is not a beauty scalar -- it is "right FOR this context." The sharpest
way to show that: hold the IMAGE perfectly fixed and change only the CONTEXT.
A context-conditional scorer must change its verdict; a per-image beauty/
confound scalar CANNOT (same image -> same score), so it scores exactly 0.5
by construction. No image editing, so no generative artifacts.

A benchmark record is a minimal pair:
    (image, context_good, context_bad)

Two data sources, both with ground truth that does NOT come from the model we
test:

  1. RUNWAY captions (data/processed/runway_captions.jsonl) -- full looks;
     garment cue -> season/formality via a transparent rules table.
  2. FASHION-PRODUCT-IMAGES (44k, parquet) -- single garments with EXPLICIT
     structured labels `usage` (Formal/Sports/Ethnic) and `articleType`
     (weather-signalling). This is real metadata, not keyword guessing, and
     gives thousands of clean pairs.

Product images are referenced as ``fpi:{id}`` and decoded from the parquet at
eval time (see fpi_loader.py) -- no thousands of files extracted to disk.

    python experiments/build_context_benchmark.py \
        --out experiments/out/context_benchmark.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# ---------- axis contexts ------------------------------------------------- #
SEASON = dict(good="a cold winter day", bad="a hot summer beach day")
FORMAL = dict(good="a formal evening event", bad="a gym workout")
ETHNIC = dict(good="a traditional festival or wedding", bad="a gym workout")


# ---------- source 1: runway captions (rules over caption text) ----------- #
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
    for line in open(path):
        c = json.loads(line)
        t = c["caption"].lower()
        img = c.get("image_path", "")
        cold, warm = _cue(t, COLD_CUES), _cue(t, WARM_CUES)
        formal, casual = _cue(t, FORMAL_CUES), _cue(t, CASUAL_CUES)
        if cold and not warm:
            out.append(dict(axis="season", source="runway", image=img,
                            cue=cold, **SEASON)); n += 1
        elif warm and not cold:
            out.append(dict(axis="season", source="runway", image=img, cue=warm,
                            good=SEASON["bad"], bad=SEASON["good"])); n += 1
        if formal and not casual:
            out.append(dict(axis="formality", source="runway", image=img,
                            cue=formal, **FORMAL)); n += 1
        elif casual and not formal:
            out.append(dict(axis="formality", source="runway", image=img,
                            cue=casual, good=FORMAL["bad"], bad=FORMAL["good"]))
            n += 1
    return n


# ---------- source 2: fashion-product-images (structured labels) ---------- #
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
        t = pq.read_table(f, columns=cols).to_pylist()
        for r in t:
            if r["masterCategory"] not in ("Apparel", "Footwear"):
                continue
            ref = f"fpi:{r['id']}"
            name = r.get("productDisplayName") or ""
            atype, usage = r["articleType"], r["usage"]

            # formality (explicit label)
            if usage == "Formal":
                out.append(dict(axis="formality", source="fpi", image=ref,
                                cue=f"usage=Formal/{atype}", **FORMAL)); n += 1
            elif usage == "Sports":
                out.append(dict(axis="formality", source="fpi", image=ref,
                                cue=f"usage=Sports/{atype}",
                                good=FORMAL["bad"], bad=FORMAL["good"])); n += 1
            elif usage == "Ethnic":
                out.append(dict(axis="occasion", source="fpi", image=ref,
                                cue=f"usage=Ethnic/{atype}", **ETHNIC)); n += 1

            # season (weather-signalling article type)
            if atype in COLD_ARTICLES:
                out.append(dict(axis="season", source="fpi", image=ref,
                                cue=f"article={atype}", **SEASON)); n += 1
            elif atype in WARM_ARTICLES:
                out.append(dict(axis="season", source="fpi", image=ref,
                                cue=f"article={atype}",
                                good=SEASON["bad"], bad=SEASON["good"])); n += 1
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

    # normalise to context_good/context_bad key names for eval
    for r in records:
        r["context_good"] = r.pop("good")
        r["context_bad"] = r.pop("bad")

    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    from collections import Counter
    by_axis = Counter(r["axis"] for r in records)
    by_src = Counter(r["source"] for r in records)
    print(f"runway pairs: {n_rw}   product pairs: {n_fpi}")
    print(f"TOTAL minimal pairs: {len(records)}")
    print(f"  by axis:   {dict(by_axis)}")
    print(f"  by source: {dict(by_src)}")
    print(f"wrote {args.out}")
    print("\nA per-image beauty/confound scalar scores BOTH contexts identically")
    print("-> 0.500 by construction. Any accuracy above 0.5 is context sensitivity")
    print("a scalar model cannot have. Score with eval_context.py (on the M4).")


if __name__ == "__main__":
    main()
