"""Grounding evaluation — designer top-k accuracy for the runway linker.

Does image↔image linking actually recover the right designer? Hold out a slice
of runway looks, predict each one's designer by k-NN vote over the rest, and
report top-1/3/5 accuracy against the random baseline. Pure numpy on the built
index (no model, no training) — a clean, reproducible thesis metric.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


def evaluate_designer_topk(
    index,
    ks: tuple[int, ...] = (1, 3, 5),
    holdout_frac: float = 0.2,
    neighbors: int = 10,
    seed: int = 42,
    split_by: str = "image",
) -> dict:
    """Runs held-out designer-prediction accuracy over a runway ``VisualIndex``.

    For each test look, retrieves its nearest train looks, aggregates their
    designers by similarity, and checks whether the true designer is in the
    top-k predicted designers.

    Args:
        index: A runway ``VisualIndex`` (metadata must have ``designer``).
        ks: Cutoffs to report accuracy at.
        holdout_frac: Fraction held out for testing.
        neighbors: Neighbours to vote over per test look.
        seed: RNG seed for the split.
        split_by: ``"image"`` (random per-image; leaky — test looks share
            collections with train) or ``"collection"`` (hold out whole shows;
            the honest cross-collection metric — can you ID a designer from an
            *unseen* collection?).

    Returns:
        ``{"top1", "top3", "top5", "n_test", "n_designers", "random_top1",
        "split_by"}``.

    Raises:
        ValueError: If the index is too small to split.
    """
    n = len(index)
    if n < 5:
        raise ValueError(f"Index too small to evaluate ({n} items).")

    designers = np.array([m.get("designer", "?") for m in index.metadata])
    n_designers = len(set(designers.tolist()))
    rng = np.random.default_rng(seed)

    if split_by == "collection":
        # Hold out whole shows so no test look shares a collection with train.
        groups = np.array([
            f"{m.get('designer','?')}|{m.get('show', m.get('season',''))}"
            for m in index.metadata
        ])
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        n_hold = max(1, int(len(uniq) * holdout_frac))
        held = set(uniq[:n_hold].tolist())
        mask = np.array([g in held for g in groups])
        test_idx = np.where(mask)[0]
        train_idx = np.where(~mask)[0]
    else:
        perm = rng.permutation(n)
        n_test = max(1, int(n * holdout_frac))
        test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_emb = index.embeddings[train_idx]           # already L2-normalised
    train_des = designers[train_idx]

    correct = {k: 0 for k in ks}
    for ti in test_idx:
        q = index.embeddings[ti]
        sims = train_emb @ q
        top = np.argsort(-sims)[:neighbors]
        agg: dict[str, float] = defaultdict(float)
        for j in top:
            agg[train_des[j]] += float(sims[j])
        ranked = [d for d, _ in sorted(agg.items(), key=lambda kv: -kv[1])]
        true = designers[ti]
        for k in ks:
            if true in ranked[:k]:
                correct[k] += 1

    n_test = len(test_idx)
    out = {f"top{k}": round(correct[k] / n_test, 3) for k in ks}
    out.update({
        "n_test": int(n_test),
        "n_designers": n_designers,
        "random_top1": round(1.0 / n_designers, 3),
        "split_by": split_by,
    })
    logger.info("Runway grounding eval: %s", out)
    return out
