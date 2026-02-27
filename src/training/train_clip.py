"""Phase 1: Fine-tuning CLIP on fashion data.

Run locally or on Google Colab A100::

    python -m src.training.train_clip
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataloader import DeepFashion2Dataset
from src.models.clip_encoder import FashionCLIPEncoder, FashionContrastiveLoss

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "embed_dim": 512,
    "freeze_backbone": True,
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-4,
    "data_root": "data/raw/deepfashion2",
    "checkpoint_dir": "data/embeddings",
    "patience": 3,
    "resume_from": None,
}


def _resolve_device() -> torch.device:
    """Selects the best available device (CUDA → CPU).

    Returns:
        A ``torch.device`` instance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)
    return device


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: FashionCLIPEncoder,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    best_loss: float,
    patience_counter: int,
) -> None:
    """Persists a full training checkpoint.

    Args:
        path: Destination file path.
        epoch: Current epoch number (0-based).
        model: The model whose state dict to save.
        optimizer: The optimizer whose state dict to save.
        scheduler: The LR scheduler whose state dict to save.
        best_loss: Best validation/train loss observed so far.
        patience_counter: Current early-stopping patience counter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            "patience_counter": patience_counter,
        },
        path,
    )
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch + 1)


def _load_checkpoint(
    path: Path,
    *,
    model: FashionCLIPEncoder,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> tuple[int, float, int]:
    """Restores training state from a checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state into.
        scheduler: Scheduler to restore state into.
        device: Device to map tensors to.

    Returns:
        A tuple ``(start_epoch, best_loss, patience_counter)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint: dict[str, Any] = torch.load(
        path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch: int = checkpoint["epoch"] + 1
    best_loss: float = checkpoint["best_loss"]
    patience_counter: int = checkpoint.get("patience_counter", 0)

    logger.info(
        "Resumed from checkpoint %s — continuing from epoch %d "
        "(best_loss=%.4f, patience=%d)",
        path,
        start_epoch + 1,
        best_loss,
        patience_counter,
    )
    return start_epoch, best_loss, patience_counter


def train(config: Optional[dict[str, Any]] = None) -> None:
    """Runs the CLIP fine-tuning loop.

    Supports **early stopping** (controlled by ``config["patience"]``) and
    **checkpoint resumption** (set ``config["resume_from"]`` to a path).

    Args:
        config: Training hyper-parameters. Missing keys are filled from
            :data:`DEFAULT_CONFIG`.

    Raises:
        RuntimeError: On unrecoverable training errors.
    """
    cfg: dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
    device: torch.device = _resolve_device()

    checkpoint_dir: Path = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path: Path = checkpoint_dir / "fashion_clip_best.pt"
    last_checkpoint_path: Path = checkpoint_dir / "fashion_clip_last_ckpt.pt"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model: FashionCLIPEncoder = FashionCLIPEncoder(
        embed_dim=cfg["embed_dim"],
        freeze_backbone=cfg["freeze_backbone"],
    ).to(device)

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    try:
        dataset: DeepFashion2Dataset = DeepFashion2Dataset(
            root=cfg["data_root"],
            split="train",
            transform=model.preprocess,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Dataset initialisation failed: %s", exc)
        raise RuntimeError("Cannot load training data.") from exc

    loader: DataLoader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer: AdamW = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=0.01,
    )
    scheduler: CosineAnnealingLR = CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    criterion: FashionContrastiveLoss = FashionContrastiveLoss(temperature=0.07)

    # ------------------------------------------------------------------
    # Optionally resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch: int = 0
    best_loss: float = float("inf")
    patience_counter: int = 0
    patience: int = cfg["patience"]

    if cfg["resume_from"] is not None:
        resume_path: Path = Path(cfg["resume_from"])
        start_epoch, best_loss, patience_counter = _load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        total_loss: float = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for batch in progress:
            images: torch.Tensor = batch["image"].to(device)
            texts: list[str] = batch["text"]

            img_emb, txt_emb = model(images, texts)
            loss: torch.Tensor = criterion(img_emb, txt_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss: float = total_loss / len(loader)
        scheduler.step()
        logger.info("Epoch %d/%d — avg_loss=%.4f", epoch + 1, cfg["epochs"], avg_loss)

        # Save last checkpoint (always)
        _save_checkpoint(
            last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_loss,
            patience_counter=patience_counter,
        )

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model.save(best_model_path)
            logger.info("  ↳ New best model saved (loss=%.4f)", best_loss)
        else:
            patience_counter += 1
            logger.info(
                "  ↳ No improvement (%d/%d patience)", patience_counter, patience
            )

        # Early stopping
        if patience_counter >= patience:
            logger.warning(
                "Early stopping triggered after %d epochs without "
                "improvement.",
                patience,
            )
            break

    logger.info("Training finished. Best loss: %.4f", best_loss)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train()

