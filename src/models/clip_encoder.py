import torch
import torch.nn as nn
import open_clip


class FashionCLIPEncoder(nn.Module):
    """
    CLIP ViT-L/14 fine-tuned на fashion данных.
    Backbone заморожен, обучается только projection head.
    """
    def __init__(self, model_name="ViT-L-14", pretrained="openai",
                 embed_dim=512, freeze_backbone=True):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            # Размораживаем последние 2 блока visual transformer
            for block in self.model.visual.transformer.resblocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        clip_dim = self.model.visual.output_dim  # 768 для ViT-L/14
        self.fashion_head = nn.Sequential(
            nn.Linear(clip_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def encode_image(self, images):
        with torch.no_grad():
            features = self.model.encode_image(images)
        return self.fashion_head(features.float())

    def encode_text(self, texts):
        tokens = self.tokenizer(texts)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return self.fashion_head(features.float())

    def forward(self, images, texts):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(texts)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return img_emb, txt_emb


class FashionContrastiveLoss(nn.Module):
    """InfoNCE loss для image-text пар."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, img_emb, txt_emb):
        batch_size = img_emb.shape[0]
        logits = (img_emb @ txt_emb.T) / self.temperature
        labels = torch.arange(batch_size, device=img_emb.device)
        return (self.ce(logits, labels) + self.ce(logits.T, labels)) / 2
