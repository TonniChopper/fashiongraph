"""Train the aesthetic scorer from one or more preference-pair sources.

Learns "which look is better" with a Bradley-Terry / RankNet loss on a small
MLP head over frozen fashion embeddings:

    loss = -log sigmoid( score(winner) - score(loser) )

Multi-source + a proper **train/val split with early stopping**, so the number
you report is honest held-out pairwise accuracy, not training accuracy.

    # after downloading the data:
    python -m fg.training.train_aesthetic --sources surrey
    python -m fg.training.train_aesthetic --sources surrey,ava --epochs 200

The trained head exports to a numpy ``.npz`` that ``fg.vision.aesthetics.
AestheticScorer`` loads at inference.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from fg.training.pair_sources import load_sources
from fg.vision.aesthetics import AestheticScorer

logger: logging.Logger = logging.getLogger(__name__)


def train(
    sources: list[str] | None = None,
    epochs: int = 200,
    hidden: int = 128,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    patience: int = 20,
    limit_items: int | None = None,
    max_pairs: int | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Trains and exports the aesthetic head with early stopping.

    Args:
        sources: Pair sources to pool (default ``["surrey"]``).
        epochs: Max epochs.
        hidden: Hidden layer size.
        lr: Learning rate.
        val_frac: Fraction of pairs held out for validation.
        patience: Early-stopping patience (epochs without val improvement).
        limit_items: Per-source image cap.
        max_pairs: Per-source pair cap.
        out_path: Export path; defaults to config path.

    Returns:
        Path of the exported head.

    Raises:
        RuntimeError: If torch is unavailable or no usable pairs remain.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Training needs torch: pip install torch") from exc

    from fg.vision.embedder import FashionEmbedder

    sources = sources or ["surrey"]
    data = load_sources(sources, limit_items=limit_items, max_pairs=max_pairs)
    if not data.pairs:
        raise RuntimeError("No preference pairs loaded — check the datasets.")

    # Embed every unique item once.
    ids = list(data.items)
    imgs = [data.items[i] for i in ids]
    embedder = FashionEmbedder()
    logger.info("Embedding %d unique items…", len(ids))
    embs = embedder.encode_images(imgs)
    pos = {iid: k for k, iid in enumerate(ids)}
    emb_t = torch.tensor(np.asarray(embs), dtype=torch.float32)
    dim = emb_t.shape[1]

    win = torch.tensor([pos[w] for w, _ in data.pairs], dtype=torch.long)
    lose = torch.tensor([pos[l] for _, l in data.pairs], dtype=torch.long)

    # Train/val split on pairs.
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(win), generator=g)
    n_val = max(1, int(len(win) * val_frac))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    logger.info("Pairs: %d train / %d val (dim=%d)", len(tr_i), len(val_i), dim)

    model = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def _pair_scores(idx):
        s_w = model(emb_t[win[idx]]).squeeze(-1)
        s_l = model(emb_t[lose[idx]]).squeeze(-1)
        return s_w, s_l

    best_val = float("inf")
    best_state = None
    best_val_acc = 0.0
    stale = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        s_w, s_l = _pair_scores(tr_i)
        loss = -torch.nn.functional.logsigmoid(s_w - s_l).mean()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vw, vl = _pair_scores(val_i)
            val_loss = -torch.nn.functional.logsigmoid(vw - vl).mean().item()
            val_acc = (vw > vl).float().mean().item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                        epoch + 1, epochs, loss.item(), val_loss, val_acc)

        if val_loss < best_val - 1e-4:
            best_val, best_val_acc = val_loss, val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stop at epoch %d (best val_acc=%.3f).", epoch + 1, best_val_acc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Best held-out pairwise accuracy: %.3f", best_val_acc)

    lin1, lin2 = model[0], model[2]
    scorer = AestheticScorer(
        w1=lin1.weight.detach().cpu().numpy().T,
        b1=lin1.bias.detach().cpu().numpy(),
        w2=lin2.weight.detach().cpu().numpy().reshape(-1),
        b2=lin2.bias.detach().cpu().numpy(),
    )
    path = scorer.save(out_path)
    logger.info("Saved aesthetic head → %s", path)
    return path


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train the aesthetic scorer.")
    p.add_argument("--sources", default="surrey", help="Comma-separated: surrey,ava")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--limit-items", type=int, default=None, help="Per-source image cap")
    p.add_argument("--max-pairs", type=int, default=None, help="Per-source pair cap")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    train(
        sources=[s.strip() for s in args.sources.split(",") if s.strip()],
        epochs=args.epochs, hidden=args.hidden, lr=args.lr,
        val_frac=args.val_frac, patience=args.patience,
        limit_items=args.limit_items, max_pairs=args.max_pairs, out_path=args.out,
    )


if __name__ == "__main__":
    main()
