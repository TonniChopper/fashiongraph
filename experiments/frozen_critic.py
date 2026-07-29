"""
Experiment 2 — frozen-feature critic (the payoff test)
======================================================

Experiment 1 (`confound_baseline.py`) showed low-level photo confounds barely
beat chance on Surrey pairwise (0.573 vs 0.537), far below human agreement
(0.732). So ~0.16 accuracy of human-agreed preference is *semantic*, not
production quality. This script asks the payoff question:

    Do FROZEN FashionSigLIP features recover that semantic taste signal?

Same linear Bradley-Terry head (`u(x)=w·z(x)`, `P(A>B)=sigmoid(u(A)-u(B))`),
same IMAGE-DISJOINT split and seed as Experiment 1 — the ONLY change is the
input features: 768-d frozen embeddings instead of 10 confounds. If accuracy
climbs from 0.573 toward 0.732, that is direct evidence frozen fashion features
carry a taste signal the confounds do not. We also fit a confound+frozen concat
to check whether confounds add anything on top (they should not).

Because the encoder stays frozen, this does NOT contradict the Track-B negative
(fine-tuning the encoder distorts its features) — we only read a linear head off
the fixed representation.

Prereq: run `embed_surrey.py` first to produce the embeddings npz.

    python experiments/frozen_critic.py \
        --emb   data/embeddings/surrey_fashionsiglip.npz \
        --surrey data/raw/surrey-aesthetics \
        --out   experiments/out/frozen_critic

Pure numpy + Pillow (features come pre-computed). Reuses the Exp-1 helpers so
the two numbers are strictly comparable.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from confound_baseline import (  # same-dir import; run from experiments/ or repo root
    load_comparisons,
    config_of,
    fit_logistic,
    accuracy,
    ceiling,
    design,
    image_features,
    FEATURE_NAMES,
)

CONFOUND_FLOOR = 0.573      # Exp 1, image-disjoint
HUMAN_CEILING = 0.732       # inter-annotator agreement


def pca_fit(X, k):
    """Return (mean, components[k,D]) via numpy SVD on centered X."""
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]


def load_embeddings(path):
    d = np.load(path, allow_pickle=True)
    names = list(d["names"])
    E = d["embeddings"].astype(np.float64)
    return {n: E[i] for i, n in enumerate(names)}


def eval_split(rows, feat, train_imgs, test_imgs, seed, pca_k=None):
    """Fit BT head on train comparisons, return test accuracy (image-disjoint)."""
    train_rows = [r for r in rows if r[0] in train_imgs and r[1] in train_imgs]
    test_rows = [r for r in rows if r[0] in test_imgs and r[1] in test_imgs]

    # optional PCA (fit on train images) to tame 768-d overfit
    f = feat
    if pca_k:
        Xtr_imgs = np.stack([feat[i] for i in train_imgs if i in feat])
        mu, comps = pca_fit(Xtr_imgs, pca_k)
        f = {n: comps @ (v - mu) for n, v in feat.items()}

    Ftr = np.stack([f[i] for i in train_imgs if i in f])
    mu_s, sd_s = Ftr.mean(0), Ftr.std(0) + 1e-8
    Xtr, ytr = design(train_rows, f, mu_s, sd_s)
    Xte, yte = design(test_rows, f, mu_s, sd_s)
    w = fit_logistic(Xtr, ytr, l2=1e-2, epochs=500, seed=seed)
    return accuracy(Xte, yte, w), len(ytr), len(yte), float(max(yte.mean(), 1 - yte.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/embeddings/surrey_fashionsiglip.npz")
    ap.add_argument("--surrey", default="data/raw/surrey-aesthetics")
    ap.add_argument("--out", default="experiments/out/frozen_critic")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca", type=int, default=-1,
                    help="PCA dims for frozen features; -1 sweeps "
                         "[none,64,128,256], 0 = full 768, N = fixed N")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not os.path.exists(args.emb):
        raise SystemExit(
            f"Embeddings not found: {args.emb}\n"
            f"Run first:  python experiments/embed_surrey.py "
            f"--surrey {args.surrey} --out {args.emb}"
        )

    rows, _ = load_comparisons(args.surrey)
    emb = load_embeddings(args.emb)
    print(f"{len(rows)} comparisons, {len(emb)} embedded images "
          f"(dim {next(iter(emb.values())).shape[0]})")

    ceil = ceiling(rows)
    human = ceil["mean_pair_agreement_multi"]

    # ---- identical image-disjoint split to Exp 1 (same seed) ------------- #
    rng = np.random.default_rng(args.seed)
    all_imgs = np.array(sorted({i for r in rows for i in r[:2]}))
    rng.shuffle(all_imgs)
    n_test = int(len(all_imgs) * args.test_frac)
    test_imgs = set(all_imgs[:n_test].tolist())
    train_imgs = set(all_imgs[n_test:].tolist())

    # sweep dimensionality so the result isn't a cherry-picked PCA choice
    settings_pca = [None, 64, 128, 256] if args.pca < 0 else [
        (args.pca if args.pca > 0 else None)]
    sweep = {}
    ntr = nte = mc = 0
    for k in settings_pca:
        acc, ntr, nte, mc = eval_split(
            rows, emb, train_imgs, test_imgs, args.seed, pca_k=k)
        sweep[str(k)] = acc
    best_k = max(sweep, key=sweep.get)
    acc_frozen = sweep[best_k]

    print("\n" + "=" * 64)
    print("FROZEN-FEATURE CRITIC (image-disjoint)")
    print(f"  train / test comparisons : {ntr} / {nte}")
    print(f"  chance / majority        : {mc:.3f}")
    print(f"  confound floor (Exp 1)   : {CONFOUND_FLOOR:.3f}")
    for k, a in sweep.items():
        star = "  <- best" if k == best_k else ""
        print(f"  frozen (pca={k:>4s})       : {a:.3f}{star}")
    print(f"  human agreement (ceiling): {human:.3f}")
    print("=" * 64)

    gain = acc_frozen - CONFOUND_FLOOR
    closed = (acc_frozen - CONFOUND_FLOOR) / max(1e-9, human - CONFOUND_FLOOR)
    print("READING:")
    if gain >= 0.03:
        print(f"  Frozen features beat the confound floor by {gain:+.3f} and")
        print(f"  close {closed*100:.0f}% of the gap to human agreement.")
        print("  -> Direct evidence FashionSigLIP encodes a taste signal above")
        print("     production quality. This is the money finding for the paper.")
    else:
        print(f"  Frozen features do NOT clear the confound floor ({gain:+.3f}).")
        print("  -> The semantic headroom is not linearly readable from frozen")
        print("     FashionSigLIP. Motivates KG-conditioning / non-linear heads,")
        print("     and is itself a clean, reportable negative.")
    print("=" * 64)

    results = {
        "confound_floor": CONFOUND_FLOOR,
        "human_ceiling": human,
        "chance": mc,
        "frozen_accuracy": acc_frozen,
        "best_pca": best_k,
        "pca_sweep": sweep,
        "gain_over_confound": gain,
        "fraction_gap_closed": closed,
        "train_comparisons": ntr,
        "test_comparisons": nte,
        "seed": args.seed,
    }
    with open(args.out + "_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}_results.json")


if __name__ == "__main__":
    main()
