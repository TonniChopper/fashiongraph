"""
Confound-only baseline for fashion "taste"
==========================================

Council verdict, first experiment.

Question this script answers
----------------------------
Before we claim any model has *taste*, we must know how far you get on the
Surrey pairwise "which look is better" data using ONLY low-level image
confounds -- brightness, contrast, colourfulness, sharpness, resolution,
edge density, entropy. No semantics. No embeddings. No knowledge graph.

If a logistic Bradley-Terry model on these dumb features already reaches the
human-agreement ceiling, then any richer "taste" scorer is unfalsifiable until
it beats this floor: it could just be a photo-quality detector. If the floor is
low, there is real headroom above production value for a genuine taste signal.

Either outcome is publishable. This script produces the number that decides
which paper we are writing.

Design notes
------------
* Utility model  u(img) = w . z(features(img))   (linear -> Bradley-Terry).
  P(A beats B) = sigmoid(u(A) - u(B)). Fit by logistic loss on feature
  DIFFERENCES, no intercept, so the model is exactly antisymmetric: swapping
  A and B flips the prediction. That is the honest form of a per-image scorer.
* Generalisation split is IMAGE-DISJOINT: images in the test comparisons are
  never seen at train time. We also report a stricter CONFIG-DISJOINT split
  (hold out whole body-shape x top x bottom configs).
* The CEILING is the best accuracy ANY per-image scorer can reach: for each
  unordered pair, always predict its majority winner. Averaged over comparisons
  this upper-bounds every deterministic model, us included. We also report raw
  inter-annotator agreement on pairs judged in more than one set.

Dependencies: numpy, Pillow.  (Deliberately no sklearn/scipy so it runs
anywhere the repo runs.)

Usage
-----
    python experiments/confound_baseline.py \
        --surrey data/raw/surrey-aesthetics \
        --out    experiments/out/confound_baseline

Writes results.json + a cached features.npz next to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
#  Feature extraction  (low-level confounds ONLY)                             #
# --------------------------------------------------------------------------- #
FEATURE_NAMES = [
    "brightness",     # mean luminance
    "contrast",       # std luminance
    "colorfulness",   # Hasler-Susstrunk
    "saturation",     # mean HSV S
    "sharpness",      # variance of Laplacian (focus / production quality)
    "edge_density",   # fraction of strong gradients
    "entropy",        # luminance histogram entropy
    "log_area",       # log(w*h)  -> resolution
    "aspect",         # h / w
    "warmth",         # mean(R-B)  colour temperature
]

_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def _lum(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _conv2d_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Tiny valid-mode 2-D convolution with a 3x3 kernel (numpy only)."""
    kh, kw = k.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * img[i:i + out.shape[0], j:j + out.shape[1]]
    return out


def image_features(path: str, work: int = 256) -> np.ndarray:
    """Return the FEATURE_NAMES vector for one image. Pure confounds."""
    im = Image.open(path).convert("RGB")
    w0, h0 = im.size                                   # true resolution
    # downscale for texture stats (keeps it fast + scale-robust)
    scale = work / max(w0, h0)
    if scale < 1.0:
        im_s = im.resize((max(1, int(w0 * scale)), max(1, int(h0 * scale))))
    else:
        im_s = im
    rgb = np.asarray(im_s, dtype=np.float64) / 255.0
    lum = _lum(rgb)

    # colourfulness (Hasler & Susstrunk 2003)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                     + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    # saturation via HSV
    sat = float(np.asarray(im_s.convert("HSV"), dtype=np.float64)[..., 1].mean() / 255.0)

    # sharpness: variance of Laplacian
    lap = _conv2d_valid(lum, _LAP)
    sharp = float(lap.var())

    # edge density
    gy, gx = np.gradient(lum)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((grad > 0.08).mean())

    # entropy of luminance histogram
    hist, _ = np.histogram(lum, bins=64, range=(0.0, 1.0), density=True)
    p = hist / (hist.sum() + 1e-12)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    return np.array([
        float(lum.mean()),
        float(lum.std()),
        colorful,
        sat,
        sharp,
        edge_density,
        entropy,
        float(np.log(w0 * h0)),
        float(h0 / max(1, w0)),
        float((R - B).mean()),
    ], dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Data loading                                                               #
# --------------------------------------------------------------------------- #
def load_comparisons(surrey_dir: str):
    """Return list of (imgA, imgB, winner) with winner in {A, B} as strings."""
    comp_dir = os.path.join(surrey_dir, "aesthetic_comparisons")
    rows = []
    files = sorted(
        f for f in os.listdir(comp_dir)
        if f.startswith("aesthetic_") and f[10:12].isdigit()
    )
    per_set = {}
    for f in files:
        s = []
        with open(os.path.join(comp_dir, f)) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                a, b, w = parts[0], parts[1], parts[2]
                win = a if w == "1" else b
                rows.append((a, b, win))
                s.append((a, b, win))
        per_set[f] = s
    return rows, per_set


def config_of(img: str) -> str:
    """'27_6.jpg' -> config '27' (body-shape x top x bottom cell)."""
    return img.split("_")[0]


# --------------------------------------------------------------------------- #
#  Numpy logistic regression on feature differences (no intercept)            #
# --------------------------------------------------------------------------- #
def fit_logistic(X, y, l2=1e-3, lr=0.5, epochs=400, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(epochs):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) / n + l2 * w
        w -= lr * grad
    return w


def accuracy(X, y, w):
    p = 1.0 / (1.0 + np.exp(-(X @ w)))
    return float(((p > 0.5).astype(int) == y).mean())


# --------------------------------------------------------------------------- #
#  Human-agreement ceiling                                                    #
# --------------------------------------------------------------------------- #
def ceiling(rows):
    """
    Best accuracy any per-image (deterministic per-pair) scorer can reach:
    for each unordered pair predict its majority winner, average over
    comparisons. Also return raw inter-judgment agreement on multiply-judged
    pairs.
    """
    votes = defaultdict(lambda: defaultdict(int))   # pair -> winner -> count
    for a, b, win in rows:
        key = tuple(sorted((a, b)))
        votes[key][win] += 1

    correct = total = 0
    multi_agree = []
    for key, vc in votes.items():
        n = sum(vc.values())
        top = max(vc.values())
        correct += top
        total += n
        if n >= 2:
            multi_agree.append(top / n)
    return {
        "ceiling_accuracy": correct / total,          # cap for any per-image model
        "n_comparisons": total,
        "n_unique_pairs": len(votes),
        "mean_pair_agreement_multi": float(np.mean(multi_agree)) if multi_agree else None,
        "n_pairs_multi_judged": len(multi_agree),
    }


# --------------------------------------------------------------------------- #
#  Build design matrix for a comparison list given per-image features         #
# --------------------------------------------------------------------------- #
def design(rows, feat, mu, sd):
    X, y = [], []
    for a, b, win in rows:
        if a not in feat or b not in feat:
            continue
        za = (feat[a] - mu) / sd
        zb = (feat[b] - mu) / sd
        X.append(za - zb)
        y.append(1 if win == a else 0)
    return np.asarray(X), np.asarray(y, dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrey", default="data/raw/surrey-aesthetics")
    ap.add_argument("--out", default="experiments/out/confound_baseline")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img_dir = os.path.join(args.surrey, "images")

    rows, per_set = load_comparisons(args.surrey)
    print(f"loaded {len(rows)} comparisons")

    # ---- features (cached) ------------------------------------------------ #
    cache = args.out + "_features.npz"
    imgs = sorted(os.listdir(img_dir))
    imgs = [f for f in imgs if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        names = list(d["names"])
        F = d["F"]
        feat = {n: F[i] for i, n in enumerate(names)}
        print(f"loaded cached features for {len(feat)} images")
    else:
        feat = {}
        t0 = time.time()
        for i, f in enumerate(imgs):
            feat[f] = image_features(os.path.join(img_dir, f))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(imgs)} images  ({time.time()-t0:.0f}s)")
        names = list(feat)
        np.savez_compressed(cache, names=np.array(names),
                            F=np.stack([feat[n] for n in names]))
        print(f"extracted + cached features for {len(feat)} images "
              f"({time.time()-t0:.0f}s)")

    # ---- ceiling ---------------------------------------------------------- #
    ceil = ceiling(rows)
    # The "predict majority winner per pair" number is INFLATED: most pairs are
    # judged once, so it scores them 100% by construction. The honest ceiling
    # for predicting a *reliable* preference is inter-annotator agreement on
    # pairs judged more than once -- that is how often two humans even agree.
    honest_ceiling = ceil["mean_pair_agreement_multi"]
    print(f"\nCeiling, naive (predict per-pair majority): "
          f"{ceil['ceiling_accuracy']:.3f}  [inflated by single-judged pairs]")
    print(f"Ceiling, HONEST (inter-annotator agreement on "
          f"{ceil['n_pairs_multi_judged']} re-judged pairs): {honest_ceiling:.3f}")

    # ---- IMAGE-DISJOINT split -------------------------------------------- #
    rng = np.random.default_rng(args.seed)
    all_imgs = np.array(sorted({i for r in rows for i in r[:2]}))
    rng.shuffle(all_imgs)
    n_test = int(len(all_imgs) * args.test_frac)
    test_imgs = set(all_imgs[:n_test].tolist())
    train_imgs = set(all_imgs[n_test:].tolist())

    train_rows = [r for r in rows if r[0] in train_imgs and r[1] in train_imgs]
    test_rows = [r for r in rows if r[0] in test_imgs and r[1] in test_imgs]

    # standardise on TRAIN images only
    Ftr = np.stack([feat[i] for i in train_imgs if i in feat])
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8

    Xtr, ytr = design(train_rows, feat, mu, sd)
    Xte, yte = design(test_rows, feat, mu, sd)
    w = fit_logistic(Xtr, ytr, seed=args.seed)
    acc_img = accuracy(Xte, yte, w)
    print(f"\n[image-disjoint]  train {len(ytr)} / test {len(yte)} comparisons")
    print(f"  confound model test accuracy : {acc_img:.3f}")
    print(f"  majority-class baseline      : {max(yte.mean(), 1-yte.mean()):.3f}")

    # ---- CONFIG-DISJOINT split (stricter) -------------------------------- #
    configs = np.array(sorted({config_of(i) for i in all_imgs}))
    rng.shuffle(configs)
    n_ctest = int(len(configs) * args.test_frac)
    ctest = set(configs[:n_ctest].tolist())
    ctrain_rows = [r for r in rows
                   if config_of(r[0]) not in ctest and config_of(r[1]) not in ctest]
    ctest_rows = [r for r in rows
                  if config_of(r[0]) in ctest and config_of(r[1]) in ctest]
    Fc = np.stack([feat[i] for i in feat if config_of(i) not in ctest])
    muc, sdc = Fc.mean(0), Fc.std(0) + 1e-8
    Xc, yc = design(ctrain_rows, feat, muc, sdc)
    Xct, yct = design(ctest_rows, feat, muc, sdc)
    wc = fit_logistic(Xc, yc, seed=args.seed)
    acc_cfg = accuracy(Xct, yct, wc) if len(yct) else None
    if acc_cfg is not None:
        print(f"\n[config-disjoint] train {len(yc)} / test {len(yct)} comparisons")
        print(f"  confound model test accuracy : {acc_cfg:.3f}")

    # ---- per-feature univariate accuracy + weights ----------------------- #
    uni = {}
    for j, name in enumerate(FEATURE_NAMES):
        wj = np.zeros(Xtr.shape[1]); wj[j] = 1.0
        # orient the single feature to whichever sign predicts better
        a = accuracy(Xte, yte, wj)
        uni[name] = max(a, 1 - a)
    weights = {n: float(v) for n, v in zip(FEATURE_NAMES, w)}
    ranked_uni = dict(sorted(uni.items(), key=lambda kv: -kv[1]))
    print("\nMost predictive single confound (univariate test acc):")
    for n, v in list(ranked_uni.items())[:5]:
        print(f"  {n:14s} {v:.3f}")

    results = {
        "n_comparisons": len(rows),
        "n_images": len(feat),
        "ceiling": ceil,
        "honest_ceiling_interannotator": honest_ceiling,
        "image_disjoint": {
            "train_comparisons": len(ytr),
            "test_comparisons": len(yte),
            "confound_accuracy": acc_img,
            "majority_baseline": float(max(yte.mean(), 1 - yte.mean())),
            "headroom_confound_to_human": honest_ceiling - acc_img,
        },
        "config_disjoint": {
            "train_comparisons": len(yc),
            "test_comparisons": len(yct),
            "confound_accuracy": acc_cfg,
        },
        "feature_weights": weights,
        "univariate_test_accuracy": ranked_uni,
        "features": FEATURE_NAMES,
    }
    with open(args.out + "_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}_results.json")

    # ---- verdict ---------------------------------------------------------- #
    c = honest_ceiling                       # read against HUMAN agreement
    mc = float(max(yte.mean(), 1 - yte.mean()))
    gap = c - acc_img
    lift = acc_img - mc                       # what confounds add over guessing
    print("\n" + "=" * 64)
    print("READING (confound floor vs human ceiling):")
    print(f"  chance / majority : {mc:.3f}")
    print(f"  confound model    : {acc_img:.3f}   (+{lift:.3f} over chance)")
    print(f"  human agreement   : {c:.3f}")
    if acc_img >= c - 0.03:
        print("  -> Confounds REACH human agreement. On this data 'taste' is")
        print("     mostly low-level photo quality. Any taste model must beat")
        print("     this floor to make a falsifiable claim.")
    elif lift <= 0.05:
        print("  -> Confounds barely beat chance and sit far below human")
        print("     agreement. Low-level photo quality does NOT explain Surrey")
        print("     preference: most of the human-agreed signal is semantic.")
        print("     Real headroom for a genuine taste model: "
              f"{gap:.3f} accuracy points.")
    else:
        print("  -> Confounds add real signal but leave headroom "
              f"({gap:.3f}) to human agreement.")
    print("  CAVEAT: Surrey is a CONTROLLED body-shape set -- it deliberately")
    print("  lacks brand/luxury/runway-photography confounds. A low confound")
    print("  floor here does NOT clear the runway set; that needs its own")
    print("  confound-controlled (minimal-pair) eval. See README.")
    print("=" * 64)


if __name__ == "__main__":
    main()
