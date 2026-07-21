"""Garment segmentation — split an outfit photo into its pieces.

Uses ``sayeed99/segformer_b3_clothes`` (SegFormer) to label clothing regions,
so one photo becomes per-garment crops + labels. This is the "segment before
embedding" idea from the ashleyashok reference: cleaner per-item signals for
Look Analysis / Personal Stylist.

Transformers/torch are imported lazily; construction fails loudly with an
install hint if they're missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fg.config import settings

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

logger: logging.Logger = logging.getLogger(__name__)

#: SegFormer labels that are actual garments/accessories (drop body parts).
GARMENT_LABELS: frozenset[str] = frozenset({
    "Hat", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt",
    "Left-shoe", "Right-shoe", "Bag", "Scarf",
})


@dataclass
class GarmentRegion:
    """One detected garment region.

    Attributes:
        label: Garment label (e.g. ``"Upper-clothes"``).
        area_fraction: Share of the image the region covers (0–1).
        box: Bounding box ``(left, top, right, bottom)``.
        crop: The cropped PIL image for this region.
    """

    label: str
    area_fraction: float
    box: tuple[int, int, int, int]
    crop: Any  # PIL.Image.Image


class GarmentSegmenter:
    """Segments outfit photos into garment regions.

    Attributes:
        model_name: HF SegFormer model id.
        device: Torch device string.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        """Loads the SegFormer model + processor.

        Args:
            model_name: HF model id; defaults to ``settings.seg_model``.
            device: Device string; auto-resolved if ``None``.

        Raises:
            RuntimeError: If transformers/torch or the model can't be loaded.
        """
        from fg.vision.embedder import resolve_device

        self.model_name = model_name or settings.seg_model
        self.device = device or resolve_device(settings.embed_device)
        try:
            import torch
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )

            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = (
                SegformerForSemanticSegmentation.from_pretrained(self.model_name)
                .to(self.device)
                .eval()
            )
            self._torch = torch
            self._id2label: dict[int, str] = self.model.config.id2label
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load segmenter '{self.model_name}'. "
                f"Install: pip install transformers torch. ({exc})"
            ) from exc
        logger.info("GarmentSegmenter '%s' on %s", self.model_name, self.device)

    def segment(self, image: "Image", min_area: float = 0.02) -> list[GarmentRegion]:
        """Segments *image* into garment regions.

        Args:
            image: A PIL image.
            min_area: Minimum area fraction to keep a region (filters specks).

        Returns:
            Garment regions sorted by descending area (largest first).
        """
        torch = self._torch
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, C, h, w)
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        seg = upsampled.argmax(dim=1)[0].cpu().numpy()

        regions = self._regions_from_map(seg, image, min_area)
        regions.sort(key=lambda r: r.area_fraction, reverse=True)
        return regions

    def _regions_from_map(self, seg, image: "Image", min_area: float) -> list[GarmentRegion]:
        """Extracts garment regions from a per-pixel label map."""
        import numpy as np

        total = seg.size
        regions: list[GarmentRegion] = []
        for label_id in np.unique(seg):
            label = self._id2label.get(int(label_id), str(label_id))
            if label not in GARMENT_LABELS:
                continue
            mask = seg == label_id
            area = float(mask.sum()) / total
            if area < min_area:
                continue
            ys, xs = np.where(mask)
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            regions.append(GarmentRegion(label, area, box, image.crop(box)))
        return regions

    def labels(self, image: "Image", min_area: float = 0.02) -> list[str]:
        """Returns just the garment labels present in *image*."""
        return [r.label for r in self.segment(image, min_area=min_area)]
