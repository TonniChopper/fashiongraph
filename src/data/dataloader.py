"""Data loaders for DeepFashion2 and trend annotation datasets."""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger: logging.Logger = logging.getLogger(__name__)


class DeepFashion2Dataset(Dataset):
    """PyTorch Dataset for DeepFashion2.

    Expected directory layout::

        <root>/
            images/                   — fashion photographs
            train_annotations.csv     — columns: image_path, category, attributes
            val_annotations.csv
            ...

    Attributes:
        root: Base directory containing images and annotation CSVs.
        transform: Optional image transform applied on loading.
        data: DataFrame of annotations for the requested split.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Any] = None,
    ) -> None:
        """Initializes the DeepFashion2 dataset.

        Args:
            root: Path to the dataset root directory.
            split: Dataset split name (used to locate
                ``{split}_annotations.csv``).
            transform: Callable image transform (e.g. from ``torchvision`` or
                OpenCLIP preprocessing).

        Raises:
            FileNotFoundError: If the annotation CSV does not exist.
            ValueError: If the CSV is empty or lacks required columns.
        """
        self.root: Path = Path(root)
        self.transform = transform

        csv_path: Path = self.root / f"{split}_annotations.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {csv_path}. "
                f"Make sure the '{split}' split CSV exists in {self.root}."
            )

        try:
            self.data: pd.DataFrame = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Failed to read annotation CSV %s: %s", csv_path, exc)
            raise

        required_columns: set[str] = {"image_path", "category", "attributes"}
        missing: set[str] = required_columns - set(self.data.columns)
        if missing:
            raise ValueError(
                f"Annotation CSV is missing required columns: {missing}"
            )

        if self.data.empty:
            raise ValueError(f"Annotation CSV is empty: {csv_path}")

        logger.info(
            "DeepFashion2Dataset loaded: split='%s', samples=%d",
            split,
            len(self.data),
        )

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Returns a single sample by index.

        Args:
            idx: Sample index.

        Returns:
            A dictionary with keys ``"image"`` (PIL Image or transformed
            tensor), ``"text"`` (category + attributes string), and
            ``"category"``.

        Raises:
            FileNotFoundError: If the image file does not exist.
            OSError: If the image cannot be opened.
        """
        row: pd.Series = self.data.iloc[idx]
        image_path: Path = self.root / "images" / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path} (sample index {idx})"
            )

        try:
            image: Image.Image = Image.open(image_path).convert("RGB")
        except OSError as exc:
            logger.error("Cannot open image %s: %s", image_path, exc)
            raise

        if self.transform:
            image = self.transform(image)

        text: str = f"{row['category']}: {row['attributes']}"
        return {"image": image, "text": text, "category": row["category"]}


class TrendAnnotationDataset(Dataset):
    """Dataset of seed trend annotations for the Temporal Fashion GNN.

    Each row in the CSV represents a trend score for one fashion element in a
    given year/season.

    Expected CSV columns: ``element``, ``category``, ``season``, ``year``,
    ``trend_score``, ``context``.

    Attributes:
        data: DataFrame with trend annotations.
        elements: Mapping from element name to a unique integer index.
        seasons: Sorted list of unique years present in the data.
    """

    def __init__(self, csv_path: str | Path) -> None:
        """Initializes the TrendAnnotationDataset.

        Args:
            csv_path: Path to the trend annotations CSV.

        Raises:
            FileNotFoundError: If *csv_path* does not exist.
            ValueError: If the CSV is empty or missing required columns.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Trend annotation file not found: {csv_path}"
            )

        try:
            self.data: pd.DataFrame = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Failed to read trend CSV %s: %s", csv_path, exc)
            raise

        required_columns: set[str] = {"element", "year", "trend_score"}
        missing: set[str] = required_columns - set(self.data.columns)
        if missing:
            raise ValueError(
                f"Trend CSV is missing required columns: {missing}"
            )

        if self.data.empty:
            raise ValueError(f"Trend annotation CSV is empty: {csv_path}")

        self.elements: dict[str, int] = {
            e: i for i, e in enumerate(self.data["element"].unique())
        }
        self.seasons: list[int] = sorted(self.data["year"].unique())

        logger.info(
            "TrendAnnotationDataset loaded: elements=%d, seasons=%d",
            len(self.elements),
            len(self.seasons),
        )

    def __len__(self) -> int:
        """Returns the number of rows in the annotation data."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Returns a single annotation row as a dictionary.

        Args:
            idx: Row index.

        Returns:
            A dictionary with the row's fields.
        """
        return dict(self.data.iloc[idx])

    def get_graph_snapshots(self) -> list[dict[str, Any]]:
        """Builds temporal graph snapshots from the annotation data.

        Each snapshot contains the trend scores of all known elements for a
        single year, suitable for consumption by
        :class:`~src.models.temporal_gnn.TemporalFashionGNN`.

        Returns:
            A list of dictionaries, each with keys ``"year"`` (int) and
            ``"scores"`` (``torch.Tensor`` of shape ``(num_elements,)``).
        """
        snapshots: list[dict[str, Any]] = []
        for year in self.seasons:
            season_data: pd.DataFrame = self.data[self.data["year"] == year]
            scores: torch.Tensor = torch.zeros(len(self.elements))
            for _, row in season_data.iterrows():
                element_idx: int = self.elements[row["element"]]
                scores[element_idx] = float(row["trend_score"])
            snapshots.append({"year": year, "scores": scores})

        logger.debug("Built %d graph snapshots.", len(snapshots))
        return snapshots

