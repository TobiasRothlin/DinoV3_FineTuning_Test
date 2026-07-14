import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


from Config import Config


class DinoV3ProjectionHead(nn.Module):
    """
    A standard MLP projection head used for both DINO and iBOT tasks.
    DINOv3 uses a 3-layer MLP.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        # The final layer uses weight normalization as per standard DINO architecture
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DinoV3Pretrainer(nn.Module):
    def __init__(self, model_name=Config.base_model, out_dim=Config.output_dim):
        super().__init__()

        # 1. The Core Backbone (from transformers)
        # This handles the RoPE, SwiGLU, and Register tokens automatically
        self.backbone = AutoModel.from_pretrained(model_name)
        embed_dim = self.backbone.config.hidden_size

        # Define hidden dimensions for the heads based on the model size.
        # For a base/large model, this is typically matched to embed_dim or scaled up.
        hidden_dim = embed_dim * 2

        # 2. The DINO Head (For Global Semantic Alignment)
        # Takes the CLS token and projects it
        self.dino_head = DinoV3ProjectionHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim
        )

        # 3. The iBOT Head (For Dense Patch Reconstruction)
        # Takes the individual patch tokens and projects them
        self.ibot_head = DinoV3ProjectionHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim
        )

    def forward(self, x):
        """
        x: A batch of images (e.g., your global or local crops)
        """
        # Pass the crops through the transformer backbone
        outputs = self.backbone(pixel_values=x)

        # Hugging Face outputs typically return the last hidden state
        # Shape: [Batch, Sequence_Length, Embed_Dim]
        hidden_states = outputs.last_hidden_state

        # Extract the CLS token (always the 0th index)
        cls_tokens = hidden_states[:, 0, :]

        # Extract the Patch tokens
        # DINOv3 uses 4 register tokens. They sit between the CLS token and the patches.
        # Index 0: CLS
        # Index 1 to 4: Registers
        # Index 5 to end: Patches
        patch_tokens = hidden_states[:, 5:, :]

        # Pass through the respective projection heads
        dino_logits = self.dino_head(cls_tokens)
        ibot_logits = self.ibot_head(patch_tokens)

        return {
            "cls_features": cls_tokens,  # Needed for L_DKoleo loss
            "patch_features": patch_tokens,  # Needed for intermediate visualizations
            "dino_logits": dino_logits,  # Needed for L_DINO loss
            "ibot_logits": ibot_logits  # Needed for L_iBOT loss
        }