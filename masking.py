import torch


class iBOTMaskGenerator:
    """Generates per-sample boolean block masks aligned to the ViT patch grid.

    Following DINOv3, masking is applied only to the global crops seen by the
    student. For each sample we decide with probability ``mask_prob`` whether to
    mask at all; if so we mask a random proportion between ``ratio_min`` and
    ``ratio_max`` of the patch tokens.
    """

    def __init__(self, ratio_min=0.1, ratio_max=0.5, mask_prob=0.5):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.mask_prob = mask_prob

    @torch.no_grad()
    def __call__(self, batch_size, num_patches, device):
        """Return a boolean mask of shape (batch_size, num_patches). True == masked."""
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

        for i in range(batch_size):
            if torch.rand(1, device=device).item() > self.mask_prob:
                continue  # This sample is left fully visible.

            ratio = self.ratio_min + torch.rand(1, device=device).item() * (self.ratio_max - self.ratio_min)
            num_mask = max(1, int(round(ratio * num_patches)))

            idx = torch.randperm(num_patches, device=device)[:num_mask]
            mask[i, idx] = True

        return mask
