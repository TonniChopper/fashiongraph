import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd


class DeepFashion2Dataset(Dataset):
    """
    Dataset для DeepFashion2.
    Структура: data/raw/deepfashion2/
      images/          — фото
      train_annotations.csv  — image_path, category, attributes
    """
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.data = pd.read_csv(self.root / f"{split}_annotations.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(self.root / "images" / row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = f"{row['category']}: {row['attributes']}"
        return {"image": image, "text": text, "category": row["category"]}


class TrendAnnotationDataset(Dataset):
    """
    Dataset на твоих seed annotations для Temporal GNN.
    CSV: element, category, season, year, trend_score, context
    """
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.elements = {e: i for i, e in enumerate(self.data["element"].unique())}
        self.seasons = sorted(self.data["year"].unique())

    def get_graph_snapshots(self):
        """Возвращает список временных срезов для PyTorch Geometric."""
        snapshots = []
        for year in self.seasons:
            season_data = self.data[self.data["year"] == year]
            scores = torch.zeros(len(self.elements))
            for _, row in season_data.iterrows():
                scores[self.elements[row["element"]]] = row["trend_score"]
            snapshots.append({"year": year, "scores": scores})
        return snapshots
