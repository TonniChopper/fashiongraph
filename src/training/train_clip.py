"""
Фаза 1: Fine-tuning CLIP на fashion данных.
Запуск локально или на Google Colab A100:
  python -m src.training.train_clip
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.clip_encoder import FashionCLIPEncoder, FashionContrastiveLoss
from src.data.dataloader import DeepFashion2Dataset


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = FashionCLIPEncoder(
        embed_dim=config["embed_dim"],
        freeze_backbone=config["freeze_backbone"]
    ).to(device)

    dataset = DeepFashion2Dataset(
        root=config["data_root"], split="train",
        transform=model.preprocess
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"],
                        shuffle=True, num_workers=4, pin_memory=True)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = FashionContrastiveLoss(temperature=0.07)
    best_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images = batch["image"].to(device)
            texts  = batch["text"]
            img_emb, txt_emb = model(images, texts)
            loss = criterion(img_emb, txt_emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "data/embeddings/fashion_clip_best.pt")
            print(f"  Saved best model")


if __name__ == "__main__":
    config = {
        "embed_dim": 512,
        "freeze_backbone": True,
        "batch_size": 64,
        "epochs": 10,
        "lr": 1e-4,
        "data_root": "data/raw/deepfashion2"
    }
    train(config)
