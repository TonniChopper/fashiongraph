"""Fashion CLIP encoder with contrastive learning support."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import open_clip

logger: logging.Logger = logging.getLogger(__name__)


class FashionCLIPEncoder(nn.Module):
    """CLIP ViT-L/14 fine-tuned on fashion data.

    The backbone is frozen by default; only the projection head is trained.
    Optionally, the last two visual-transformer blocks are unfrozen for
    domain adaptation.

    Attributes:
        model: The underlying OpenCLIP model.
        preprocess: Image preprocessing transform provided by OpenCLIP.
        tokenizer: Text tokenizer provided by OpenCLIP.
        fashion_head: Learnable projection head that maps CLIP features
            to the target embedding dimension.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        embed_dim: int = 512,
        freeze_backbone: bool = True,
    ) -> None:
        """Initializes the FashionCLIPEncoder.

        Args:
            model_name: OpenCLIP model architecture name.
            pretrained: Pretrained weights identifier.
            embed_dim: Output embedding dimensionality.
            freeze_backbone: If ``True``, freeze the backbone and only unfreeze
                the last two visual-transformer blocks.

        Raises:
            RuntimeError: If the OpenCLIP model cannot be created.
        """
        super().__init__()

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except Exception as exc:
            logger.error("Failed to create OpenCLIP model '%s': %s", model_name, exc)
            raise RuntimeError(
                f"Could not create OpenCLIP model '{model_name}'"
            ) from exc

        self.tokenizer = open_clip.get_tokenizer(model_name)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze the last 2 visual-transformer blocks for domain adaptation
            for block in self.model.visual.transformer.resblocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
            logger.info(
                "Backbone frozen; last 2 visual-transformer blocks unfrozen."
            )

        clip_dim: int = self.model.visual.output_dim  # 768 for ViT-L/14
        self.fashion_head: nn.Sequential = nn.Sequential(
            nn.Linear(clip_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        logger.info(
            "FashionCLIPEncoder initialised: clip_dim=%d, embed_dim=%d",
            clip_dim,
            embed_dim,
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encodes images through the frozen backbone and fashion head.

        Args:
            images: Batch of preprocessed images, shape ``(B, C, H, W)``.

        Returns:
            Image embeddings of shape ``(B, embed_dim)``.
        """
        with torch.no_grad():
            features: torch.Tensor = self.model.encode_image(images)
        return self.fashion_head(features.float())

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encodes text descriptions through the frozen backbone and fashion head.

        Args:
            texts: List of text descriptions.

        Returns:
            Text embeddings of shape ``(B, embed_dim)``.
        """
        tokens: torch.Tensor = self.tokenizer(texts)
        with torch.no_grad():
            features: torch.Tensor = self.model.encode_text(tokens)
        return self.fashion_head(features.float())

    def forward(
        self, images: torch.Tensor, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes L2-normalised image and text embeddings.

        Args:
            images: Batch of preprocessed images, shape ``(B, C, H, W)``.
            texts: List of text descriptions (length ``B``).

        Returns:
            A tuple ``(img_emb, txt_emb)`` of L2-normalised embeddings,
            each of shape ``(B, embed_dim)``.
        """
        img_emb: torch.Tensor = self.encode_image(images)
        txt_emb: torch.Tensor = self.encode_text(texts)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return img_emb, txt_emb

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Saves the model state dict to *path*.

        Args:
            path: Destination file path (e.g. ``"checkpoints/model.pt"``).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path)
            logger.info("Model saved to %s", path)
        except OSError as exc:
            logger.error("Failed to save model to %s: %s", path, exc)
            raise

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        embed_dim: int = 512,
        freeze_backbone: bool = True,
        map_location: Any = None,
    ) -> "FashionCLIPEncoder":
        """Loads a previously saved ``FashionCLIPEncoder`` from *path*.

        Args:
            path: Path to the saved state dict.
            model_name: OpenCLIP model architecture name (must match the saved
                model).
            pretrained: Pretrained weights identifier.
            embed_dim: Output embedding dimensionality (must match the saved
                model).
            freeze_backbone: Whether the backbone should be frozen.
            map_location: ``torch.load`` map_location argument (e.g.
                ``"cpu"``).

        Returns:
            A ``FashionCLIPEncoder`` instance with restored weights.

        Raises:
            FileNotFoundError: If *path* does not exist.
            RuntimeError: If the state dict is incompatible.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        instance = cls(
            model_name=model_name,
            pretrained=pretrained,
            embed_dim=embed_dim,
            freeze_backbone=freeze_backbone,
        )
        state_dict: dict[str, Any] = torch.load(
            path, map_location=map_location, weights_only=True
        )
        instance.load_state_dict(state_dict)
        logger.info("Model loaded from %s", path)
        return instance


class FashionContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss for image-text pairs.

    Attributes:
        temperature: Softmax temperature scaling factor.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """Initializes the contrastive loss.

        Args:
            temperature: Temperature for the softmax. Lower values produce
                sharper distributions.
        """
        super().__init__()
        self.temperature: float = temperature
        self.ce: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(
        self, img_emb: torch.Tensor, txt_emb: torch.Tensor
    ) -> torch.Tensor:
        """Computes the symmetric InfoNCE loss.

        Args:
            img_emb: L2-normalised image embeddings, shape ``(B, D)``.
            txt_emb: L2-normalised text embeddings, shape ``(B, D)``.

        Returns:
            Scalar loss tensor.
        """
        batch_size: int = img_emb.shape[0]
        logits: torch.Tensor = (img_emb @ txt_emb.T) / self.temperature
        labels: torch.Tensor = torch.arange(batch_size, device=img_emb.device)
        return (self.ce(logits, labels) + self.ce(logits.T, labels)) / 2

