"""Encode Vogue Runway images with FashionCLIPEncoder and save to disk."""

import csv
import logging
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.clip_encoder import FashionCLIPEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

CSV_PATH: Path = Path("data/raw/vogue_runway/metadata.csv")
CHECKPOINT: Path = Path("checkpoints/clip_epoch3.pt")  # adjust if needed
OUTPUT: Path = Path("data/embeddings/runway_clip.pt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE: int = 32

transform: T.Compose = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------


class RunwayDataset(Dataset):
    """Simple dataset that loads runway images for CLIP encoding.

    Filters out rows whose ``local_path`` does not exist on disk.

    Attributes:
        rows: Filtered list of CSV row dicts.
        transform: Image preprocessing transform.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        img_transform: T.Compose,
    ) -> None:
        self.rows: list[dict[str, str]] = [
            r for r in rows if Path(r["local_path"]).exists()
        ]
        self.transform: T.Compose = img_transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row: dict[str, str] = self.rows[idx]
        img: Image.Image = Image.open(row["local_path"]).convert("RGB")
        return self.transform(img), idx


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Encodes all runway images with CLIP and saves embeddings + metadata."""
    if not CSV_PATH.exists():
        logger.error("Metadata CSV not found: %s", CSV_PATH)
        sys.exit(1)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load model -------------------------------------------------
    model: FashionCLIPEncoder = FashionCLIPEncoder()

    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        logger.info("Loaded checkpoint: %s", CHECKPOINT)
    else:
        logger.info("No checkpoint found at %s — using pretrained weights.", CHECKPOINT)

    model = model.to(device)
    model.eval()

    # ---- read CSV ---------------------------------------------------
    with open(CSV_PATH, encoding="utf-8") as f:
        rows: list[dict[str, str]] = list(csv.DictReader(f))

    dataset = RunwayDataset(rows, transform)
    loader: DataLoader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True,
    )
    logger.info(
        "Dataset: %d images (of %d rows). Batch size: %d.",
        len(dataset), len(rows), BATCH_SIZE,
    )

    # ---- encode -----------------------------------------------------
    embeddings: list[torch.Tensor] = []
    metadata: list[dict[str, str]] = []
    processed: int = 0

    for imgs, indices in loader:
        imgs = imgs.to(device)
        try:
            with torch.no_grad():
                emb: torch.Tensor = model.encode_image(imgs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())

            for idx in indices.tolist():
                row: dict[str, str] = dataset.rows[idx]
                metadata.append({
                    "designer": row["designer"],
                    "show": row["show"],
                    "look_index": row["look_index"],
                    "image_path": row["local_path"],
                })
        except Exception as exc:
            logger.warning("Batch failed: %s", exc)
            continue

        processed += imgs.size(0)
        if processed % 200 < BATCH_SIZE:
            logger.info("Progress: %d/%d", processed, len(dataset))

    if not embeddings:
        logger.warning("No embeddings produced. Exiting.")
        return

    torch.save({"embeddings": torch.cat(embeddings), "metadata": metadata}, OUTPUT)
    logger.info("✅ Saved %d runway embeddings to %s", len(metadata), OUTPUT)


if __name__ == "__main__":
    main()

