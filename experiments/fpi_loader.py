"""
Resolve `fpi:{id}` image references from the fashion-product-images parquet.

The benchmark stores product images as `fpi:{id}` instead of extracting tens of
thousands of files. This loads the PIL images for a given set of ids in a single
pass over the parquet shards.
"""
from __future__ import annotations

import glob
import io

from PIL import Image


def load_fpi_images(ids, pattern="data/raw/fashion-product-images-small/data/*.parquet"):
    """ids: iterable of int/str product ids -> dict {str(id): PIL.Image}."""
    import pyarrow.parquet as pq
    want = {str(i) for i in ids}
    out = {}
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f, columns=["id", "image"]).to_pylist()
        for r in t:
            sid = str(r["id"])
            if sid in want and sid not in out:
                b = r["image"]["bytes"]
                out[sid] = Image.open(io.BytesIO(b)).convert("RGB")
        if len(out) == len(want):
            break
    return out
