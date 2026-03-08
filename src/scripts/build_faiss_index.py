"""Merge Fashionpedia + Runway embeddings into a single index (no faiss)."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch, json
import numpy as np
from pathlib import Path

FASHIONPEDIA = Path("data/embeddings/fashionpedia_clip.pt")
RUNWAY       = Path("data/embeddings/runway_clip.pt")
OUT_INDEX    = Path("data/embeddings/fashion_index.pt")
OUT_META     = Path("data/embeddings/fashion_meta.json")

fp = torch.load(FASHIONPEDIA)
rw = torch.load(RUNWAY)

embs = torch.cat([fp["embeddings"], rw["embeddings"]]).float()

# L2 normalize
embs = embs / embs.norm(dim=-1, keepdim=True)

fp_meta = [{"source": "fashionpedia", "label": str(l)} for l in fp["labels"]]
meta = fp_meta + rw["metadata"]

torch.save(embs, str(OUT_INDEX))
OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

print(f"✅ Index: {embs.shape[0]} vectors | dim={embs.shape[1]}")
