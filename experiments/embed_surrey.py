"""
Embed the Surrey aesthetics images with the project's frozen FashionSigLIP.

Run this ONCE on a machine where the embedder is set up (your M4 — MPS + the
Marqo/marqo-fashionSigLIP weights are already cached from building the runway
index). It takes ~1 minute and writes a cache the taste experiments read.

    python experiments/embed_surrey.py \
        --surrey data/raw/surrey-aesthetics \
        --out    data/embeddings/surrey_fashionsiglip.npz

Output npz: names (str array, e.g. '27_6.jpg') and embeddings (N, 768) float32,
L2-normalised — identical pipeline to data/embeddings/runway_fashionsiglip.npz.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image

from fg.vision.embedder import FashionEmbedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrey", default="data/raw/surrey-aesthetics")
    ap.add_argument("--out", default="data/embeddings/surrey_fashionsiglip.npz")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    img_dir = os.path.join(args.surrey, "images")
    names = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"embedding {len(names)} images with frozen FashionSigLIP…")

    emb = FashionEmbedder()
    t0 = time.time()
    vecs = []
    for start in range(0, len(names), args.batch_size):
        batch = names[start:start + args.batch_size]
        imgs = [Image.open(os.path.join(img_dir, n)) for n in batch]
        vecs.append(emb.encode_images(imgs, batch_size=args.batch_size))
        print(f"  {min(start+args.batch_size, len(names))}/{len(names)} "
              f"({time.time()-t0:.0f}s)")
    E = np.concatenate(vecs, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), embeddings=E)
    print(f"wrote {args.out}  shape {E.shape}")


if __name__ == "__main__":
    main()
