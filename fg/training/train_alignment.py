"""Track B — KG-concept alignment of the fashion embedder (projection head).

Phase-5 council pick. We learn a projection over the **frozen** FashionSigLIP
image embeddings (already sitting in the runway index) so that a look lands
nearer the *concepts its designer is known for* — with the concepts supplied by
the knowledge graph. The image tower stays frozen, so this trains in minutes on
the M4; only if it plateaus do we escalate to unfreezing the tower on Colab.

The whole point is the **ablation**, so one script runs either arm::

    # KG-concept supervision (the thesis claim):
    python -m fg.training.train_alignment --supervision concepts
    # designer-label-only control (FashionKLIP-style):
    python -m fg.training.train_alignment --supervision labels
    # both + the base embedder, printed as one comparison table:
    python -m fg.training.train_alignment --supervision both

Protocol (honest by design):
* Collections are split three ways — **fit / val / test** — by whole show, so no
  look leaks across splits (mirrors ``runway_eval`` leakage rules).
* The projection is trained on *fit* only; early-stopped on *val* designer top-1;
  the reported number is *test* designer top-k, which training never saw.
* ``base`` (identity projection) is evaluated on the same test split, and a
  bootstrap CI is reported so a small delta isn't over-claimed.

Loss: multi-positive InfoNCE between the projected look and the frozen *text*
embeddings of its positive anchors (concepts, or the single designer for the
label arm), with every other anchor as a negative.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from fg.config import settings
from fg.kg.store import KnowledgeGraph
from fg.training.alignment_pairs import LookRecord, load_supervision
from fg.vision.index import VisualIndex

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Splitting + evaluation (pure numpy — no torch, no model)
# ---------------------------------------------------------------------------

def split_groups_three(
    records: list[LookRecord],
    test_frac: float = 0.2,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits row indices into fit/val/test by whole collection (no leakage).

    Args:
        records: Output of :func:`load_supervision`.
        test_frac: Fraction of collections held out for the final test.
        val_frac: Fraction of collections used for early-stopping.
        seed: RNG seed.

    Returns:
        ``(fit_rows, val_rows, test_rows)`` int arrays.
    """
    groups = np.array([r.group for r in records])
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_frac))
    n_val = max(1, int(len(uniq) * val_frac))
    test_g = set(uniq[:n_test].tolist())
    val_g = set(uniq[n_test:n_test + n_val].tolist())
    rows = np.array([r.row for r in records])
    is_test = np.array([g in test_g for g in groups])
    is_val = np.array([g in val_g for g in groups])
    return rows[~(is_test | is_val)], rows[is_val], rows[is_test]


def designer_topk(
    emb: np.ndarray,
    designers: np.ndarray,
    ref_rows: np.ndarray,
    query_rows: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 5),
    neighbors: int = 10,
) -> dict:
    """kNN designer-prediction accuracy for explicit reference/query splits.

    Args:
        emb: L2-normalised embeddings ``(N, D)``.
        designers: Per-row designer keys ``(N,)``.
        ref_rows: Rows used as the retrieval database.
        query_rows: Rows to predict.
        ks: Cutoffs.
        neighbors: Neighbours to vote over.

    Returns:
        ``{"top1","top3","top5","n_test","hit1"}`` — ``hit1`` is the per-query
        top-1 correctness array (for bootstrapping).
    """
    ref_emb = emb[ref_rows]
    ref_des = designers[ref_rows]
    correct = {k: 0 for k in ks}
    hit1 = np.zeros(len(query_rows), dtype=np.float64)
    for qi, r in enumerate(query_rows):
        sims = ref_emb @ emb[r]
        top = np.argsort(-sims)[:neighbors]
        agg: dict[str, float] = defaultdict(float)
        for j in top:
            agg[ref_des[j]] += float(sims[j])
        ranked = [d for d, _ in sorted(agg.items(), key=lambda kv: -kv[1])]
        true = designers[r]
        for k in ks:
            if true in ranked[:k]:
                correct[k] += 1
        hit1[qi] = 1.0 if ranked[:1] == [true] else 0.0
    n = len(query_rows)
    out = {f"top{k}": round(correct[k] / n, 3) for k in ks}
    out["n_test"] = int(n)
    out["hit1"] = hit1
    return out


def bootstrap_ci(hit1: np.ndarray, iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for a mean 0/1 accuracy array."""
    rng = np.random.default_rng(seed)
    n = len(hit1)
    if n == 0:
        return (0.0, 0.0)
    means = [hit1[rng.integers(0, n, n)].mean() for _ in range(iters)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (round(float(lo), 3), round(float(hi), 3))


# ---------------------------------------------------------------------------
# Positive-pair weights (the KG's role: who should attract whom, in image space)
# ---------------------------------------------------------------------------

def build_designer_weights(
    records: list[LookRecord], supervision: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Builds per-row designer ids and a designer×designer positive-weight matrix.

    This is where the knowledge graph enters — as the *supervision graph* for
    supervised contrastive learning, not as text anchors:

    * ``"labels"`` — a look attracts only same-designer looks (``Wdd = I``). This
      is the FashionKLIP-style control.
    * ``"concepts"`` — a look attracts same-designer looks *and*, partially, looks
      from designers with overlapping KG concepts (weight = Jaccard of the two
      designers' shared-concept sets). The graph structure decides the pull.

    Args:
        records: Output of :func:`load_supervision`.
        supervision: ``"labels"`` or ``"concepts"``.

    Returns:
        ``(row_designer_idx, Wdd, designer_keys)``.

    Raises:
        ValueError: On unknown *supervision*.
    """
    keys = sorted({r.designer for r in records})
    idx_of = {k: i for i, k in enumerate(keys)}
    row_des = np.array([idx_of[r.designer] for r in records], dtype=np.int64)
    n = len(keys)

    if supervision == "labels":
        return row_des, np.eye(n, dtype=np.float32), keys

    if supervision == "concepts":
        # Full per-designer concept sets — unique concepts count in the union, so
        # designers stay distinct (Jaccard matches the signal_report number).
        concepts_of = {
            k: set(next(r.concepts for r in records if r.designer == k))
            for k in keys
        }
        Wdd = np.eye(n, dtype=np.float32)
        for a in range(n):
            for b in range(a + 1, n):
                A, B = concepts_of[keys[a]], concepts_of[keys[b]]
                j = len(A & B) / len(A | B) if (A or B) else 0.0
                Wdd[a, b] = Wdd[b, a] = j
        return row_des, Wdd, keys

    raise ValueError(f"Unknown supervision {supervision!r} (concepts|labels).")


# ---------------------------------------------------------------------------
# Training (torch, frozen embeddings → learn a projection in *image* space)
# ---------------------------------------------------------------------------

def train_projection(
    Z: np.ndarray,
    row_des: np.ndarray,
    Wdd: np.ndarray,
    fit_rows: np.ndarray,
    val_rows: np.ndarray,
    designers: np.ndarray,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    temp: float = 0.1,
    weight_decay: float = 1e-4,
    patience: int = 40,
    seed: int = 42,
) -> np.ndarray:
    """Learns a ``(D, D)`` projection via *supervised contrastive* loss.

    Operates in the same image↔image space the metric uses: projected looks are
    pulled toward their positive looks (weighted by :func:`build_designer_weights`)
    and pushed from the rest of the batch. Initialised to identity, so training
    starts *at* the base embedder; the val-best projection is kept.

    Args:
        Z: Frozen L2-normalised image embeddings ``(N, D)``.
        row_des: Per-row designer index ``(N,)``.
        Wdd: Designer×designer positive-weight matrix.
        fit_rows: Rows to train on (full-batch — a few thousand fits on the M4).
        val_rows: Rows for early-stopping (designer top-1).
        designers: Per-row designer keys (for the val metric).
        epochs, lr, temp, weight_decay, patience, seed: Optimisation knobs.

    Returns:
        The projection ``W`` as ``(D, D)`` float32 numpy (apply as ``Z @ W.T``).

    Raises:
        RuntimeError: If torch is unavailable.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Training needs torch: pip install torch") from exc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    D = Z.shape[1]

    fit = np.asarray(fit_rows, dtype=np.int64)
    Zf = torch.tensor(Z[fit], dtype=torch.float32, device=device)
    # Row-level positive weights from the designer-level matrix, self excluded.
    des_f = row_des[fit]
    Wrow = torch.tensor(Wdd[np.ix_(des_f, des_f)], dtype=torch.float32, device=device)
    Wrow.fill_diagonal_(0.0)
    row_sum = Wrow.sum(dim=1).clamp(min=1e-8)
    eye = torch.eye(len(fit), dtype=torch.bool, device=device)

    W = torch.eye(D, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr, weight_decay=weight_decay)

    def _val_top1(mat: np.ndarray) -> float:
        emb = Z @ mat.T
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        return designer_topk(emb, designers, fit_rows, val_rows)["top1"]

    best_val = _val_top1(np.eye(D, dtype=np.float32))   # base (identity) score
    best_W = np.eye(D, dtype=np.float32)
    stale = 0

    for epoch in range(epochs):
        opt.zero_grad()
        P = Zf @ W.T
        P = P / P.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sims = (P @ P.T) / temp                          # (B, B)
        sims = sims.masked_fill(eye, torch.finfo(torch.float32).min)
        log_prob = sims - torch.logsumexp(sims, dim=1, keepdim=True)
        loss = -(Wrow * log_prob).sum(dim=1).div(row_sum).mean()
        loss.backward()
        opt.step()

        val = _val_top1(W.detach().cpu().numpy())
        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info("epoch %d/%d — loss=%.4f val_top1=%.3f",
                        epoch + 1, epochs, float(loss), val)
        if val > best_val + 1e-4:
            best_val, best_W, stale = val, W.detach().cpu().numpy().copy(), 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stop @ epoch %d (best val_top1=%.3f)", epoch + 1, best_val)
                break

    logger.info("Best validation designer top-1: %.3f", best_val)
    return best_W.astype(np.float32)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _project_all(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    emb = Z @ W.T
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


def run(
    supervision: str = "concepts",
    index_path: str | Path | None = None,
    kg_path: str | Path | None = None,
    *,
    test_frac: float = 0.2,
    val_frac: float = 0.15,
    epochs: int = 300,
    lr: float = 1e-3,
    temp: float = 0.1,
    seed: int = 42,
    save: bool = True,
) -> dict:
    """Trains one arm (or both) and returns the comparison table.

    Args:
        supervision: ``"concepts"``, ``"labels"``, or ``"both"``.
        index_path: Runway index; defaults to configured runway index.
        kg_path: KG sqlite path; defaults to configured KG.
        test_frac, val_frac: Collection-split fractions.
        epochs, lr, temp, seed: Training knobs.
        save: Whether to write the projection ``.npz`` per arm.

    Returns:
        ``{arm: {top1, top3, top5, ci95, n_test}}`` including ``"base"``.
    """
    idx = VisualIndex.load(index_path or settings.embeddings_dir / settings.runway_index_name)
    kg = KnowledgeGraph(str(kg_path) if kg_path else None)
    records = load_supervision(idx, kg)

    Z = idx.embeddings
    designers = np.array([r.designer for r in records])
    fit_rows, val_rows, test_rows = split_groups_three(records, test_frac, val_frac, seed)
    ref_rows = np.concatenate([fit_rows, val_rows])
    logger.info("Split: fit=%d val=%d test=%d (by collection)",
                len(fit_rows), len(val_rows), len(test_rows))

    # Baseline: the frozen embedder, same test split.
    base = designer_topk(Z, designers, ref_rows, test_rows)
    results: dict[str, dict] = {"base": {
        **{k: base[k] for k in ("top1", "top3", "top5", "n_test")},
        "ci95": bootstrap_ci(base["hit1"]),
    }}

    arms = ["concepts", "labels"] if supervision == "both" else [supervision]
    for arm in arms:
        row_des, Wdd, keys = build_designer_weights(records, arm)
        logger.info("[%s] %d designers; mean off-diag positive weight=%.3f",
                    arm, len(keys),
                    float((Wdd.sum() - len(keys)) / max(1, len(keys) * (len(keys) - 1))))
        W = train_projection(
            Z, row_des, Wdd, fit_rows, val_rows, designers,
            epochs=epochs, lr=lr, temp=temp, seed=seed,
        )
        emb = _project_all(Z, W)
        res = designer_topk(emb, designers, ref_rows, test_rows)
        results[arm] = {
            **{k: res[k] for k in ("top1", "top3", "top5", "n_test")},
            "ci95": bootstrap_ci(res["hit1"]),
        }
        if save:
            out = settings.embeddings_dir / f"alignment_{arm}.npz"
            np.savez(out, W=W, supervision=arm)
            logger.info("Saved projection → %s", out)

    _print_table(results)
    return results


def _print_table(results: dict[str, dict]) -> None:
    """Prints the base/labels/concepts comparison table."""
    order = [k for k in ("base", "labels", "concepts") if k in results]
    print("\n=== Designer top-k (honest by-collection test split) ===")
    print(f"{'condition':<10} {'top1':>6} {'95% CI':>14} {'top3':>6} {'top5':>6}  n")
    for k in order:
        r = results[k]
        ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
        print(f"{k:<10} {r['top1']:>6.3f} {ci:>14} {r['top3']:>6.3f} {r['top5']:>6.3f}  {r['n_test']}")
    if "concepts" in results and "labels" in results:
        d = results["concepts"]["top1"] - results["labels"]["top1"]
        print(f"\nKG-concept vs label-only Δtop1 = {d:+.3f}  "
              f"(win = positive and CIs separated)")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Track B — KG-concept embedder alignment.")
    p.add_argument("--supervision", default="both",
                   choices=["concepts", "labels", "both"],
                   help="Which arm(s) to train (default: both → full ablation).")
    p.add_argument("--index", default=None, help="Runway index .npz")
    p.add_argument("--kg", default=None, help="KG sqlite path")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    run(
        supervision=args.supervision, index_path=args.index, kg_path=args.kg,
        test_frac=args.test_frac, val_frac=args.val_frac, epochs=args.epochs,
        lr=args.lr, temp=args.temp, seed=args.seed, save=not args.no_save,
    )


if __name__ == "__main__":
    main()
