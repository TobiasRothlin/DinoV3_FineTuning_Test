import torch
import torch.nn as nn
import copy
from transformers import AutoModel


class DINOv3ForSelfSupervisedPretraining(nn.Module):
    def __init__(self, model_id="facebook/dinov3-vitl16-pretrain-lvd1689m", out_dim=65536):
        super().__init__()

        # 1. Load the core Vision Transformer backbone from Hugging Face
        self.backbone = AutoModel.from_pretrained(model_id)

        # Dynamically grab the hidden dimension size (e.g., 1024 for ViT-L)
        hidden_dim = self.backbone.config.hidden_size

        # 2. Build the Projection Head
        # The original DINOv3 uses separate heads for the global and local losses[cite: 1].
        # For your Proof of Concept, a standard MLP projection head works perfectly
        # to map the features into the higher dimensionality expected by the DINO loss.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, pixel_values):
        # Pass the image crops through the Hugging Face backbone
        outputs = self.backbone(pixel_values=pixel_values)

        # The backbone outputs a global [CLS] token alongside a grid of patch-level embeddings[cite: 1].
        # We extract the [CLS] token (the first token in the sequence at index 0)
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Pass the [CLS] token through the projection head
        projected_cls = self.head(cls_token)

        return projected_cls