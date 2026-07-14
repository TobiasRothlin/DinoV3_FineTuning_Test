import torch.nn as nn
from transformers import AutoModel


def _build_head(in_dim, hidden_dim, out_dim):
    """Standard DINO/iBOT MLP projection head."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class DINOv3ForSelfSupervisedPretraining(nn.Module):
    """DINOv3 backbone with separate DINO (global CLS) and iBOT (dense patch) heads.

    ``forward`` returns three tensors:
      * projected_cls   -> CLS token through the ``dino_head``  (global semantics)
      * projected_patch -> patch tokens through the ``ibot_head`` (dense geometries)
      * raw_patches     -> raw backbone patch tokens (bypass heads, for Gram anchoring)
    """

    def __init__(
        self,
        model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        dino_out_dim=65536,
        ibot_out_dim=65536,
        hidden_dim=2048,
    ):
        super().__init__()

        # 1. Load the core Vision Transformer backbone from Hugging Face
        self.backbone = AutoModel.from_pretrained(model_id)

        # Dynamically grab the hidden dimension size (e.g., 1024 for ViT-L)
        embed_dim = self.backbone.config.hidden_size

        # 2. Two independent projection heads. DINOv3 uses a dedicated head for the
        #    global DINO objective and another for the dense iBOT objective.
        self.dino_head = _build_head(embed_dim, hidden_dim, dino_out_dim)
        self.ibot_head = _build_head(embed_dim, hidden_dim, ibot_out_dim)

        # NOTE: The DINOv3 backbone already owns a learnable ``mask_token`` inside its
        # embeddings and applies it natively when ``bool_masked_pos`` is passed, so we
        # do not add our own.

        # Number of special (non-patch) prefix tokens: CLS + register tokens.
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)

    def _num_prefix_tokens(self):
        # CLS token (index 0) + register tokens precede the patch grid.
        return 1 + self.num_register_tokens

    def forward(self, pixel_values, mask=None):
        """Run the backbone and both heads.

        Args:
            pixel_values: (B, 3, H, W) image batch.
            mask: optional boolean tensor (B, num_patches) where True marks a masked
                  patch. Forwarded to the backbone as ``bool_masked_pos`` so it swaps
                  in its native ``mask_token`` at those positions.

        Returns:
            projected_cls:   (B, dino_out_dim)
            projected_patch: (B, num_patches, ibot_out_dim)
            raw_patches:     (B, num_patches, embed_dim)
        """
        outputs = self.backbone(pixel_values=pixel_values, bool_masked_pos=mask)

        last_hidden = outputs.last_hidden_state
        prefix = self._num_prefix_tokens()

        # Slice the sequence: CLS token (index 0) and the patch tokens (after prefix).
        cls_token = last_hidden[:, 0, :]
        raw_patches = last_hidden[:, prefix:, :]

        projected_cls = self.dino_head(cls_token)
        projected_patch = self.ibot_head(raw_patches)

        return projected_cls, projected_patch, raw_patches

