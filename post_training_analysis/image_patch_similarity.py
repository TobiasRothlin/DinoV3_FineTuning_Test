import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or try 'Qt5Agg' if you have PyQt installed
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoModel


# Fine-tuned model: custom normalization computed on the training data
MEAN = [0.4114, 0.4183, 0.4359]
STD = [0.3005, 0.2981, 0.2966]
# Base model: standard ImageNet normalization (original DINOv3 pre-training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TargetSize = (2304, 832)
Model = r"C:\Users\TobiasRothlin\Downloads\dinov3-vitl16-finetuned-v1-SyntoGo\dinov3-vitl16-finetuned-v1-SyntoGo"
BaseModel = "facebook/dinov3-vitl16-pretrain-lvd1689m"

class DinoInteractiveExplorer:
    def __init__(self, image_path, model_id=Model, base_model_id=BaseModel, target_size=TargetSize):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing DINOv3 Explorer on {self.device}...")

        self.patch_size = 16
        self.target_size = target_size
        self.raw_image = Image.open(image_path).convert("RGB")
        self.orig_W, self.orig_H = self.raw_image.size

        # 1. Load both models: the fine-tuned model and the original base model
        print(f"Loading fine-tuned model: {model_id}")
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        print(f"Loading base model: {base_model_id}")
        self.base_model = AutoModel.from_pretrained(base_model_id).to(self.device)
        self.base_model.eval()

        # 2. Extract and cache patch embeddings for both models.
        #    The fine-tuned model uses its custom normalization; the base model
        #    uses standard ImageNet normalization.
        print("Extracting features (this only happens once)...")
        self.patch_embs = self._extract_patch_embeddings(self.model, MEAN, STD)
        self.base_patch_embs = self._extract_patch_embeddings(self.base_model, IMAGENET_MEAN, IMAGENET_STD)

        # 3. Setup Interactive UI (image + fine-tuned map + base map)
        self.fig, self.axes = plt.subplots(1, 3, figsize=(22, 8))
        self.fig.canvas.manager.set_window_title('DINOv3 Interactive Patch Similarity (Fine-tuned vs Base)')

        # Left Axis: Clickable Original Image
        self.axes[0].imshow(self.raw_image)
        self.axes[0].set_title("Click anywhere to select a patch")
        self.axes[0].axis("off")

        # Middle Axis: Fine-tuned model heatmap overlay
        self.axes[1].imshow(self.raw_image)
        self.heatmap_overlay = self.axes[1].imshow(
            np.zeros((self.orig_H, self.orig_W)),
            cmap='jet',
            alpha=0.6,
            vmin=0.0, vmax=1.0
        )
        self.axes[1].set_title("Fine-tuned model similarity")
        self.axes[1].axis("off")
        self.cbar = self.fig.colorbar(self.heatmap_overlay, ax=self.axes[1], fraction=0.046, pad=0.04)
        self.cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)

        # Right Axis: Base model heatmap overlay
        self.axes[2].imshow(self.raw_image)
        self.base_heatmap_overlay = self.axes[2].imshow(
            np.zeros((self.orig_H, self.orig_W)),
            cmap='jet',
            alpha=0.6,
            vmin=0.0, vmax=1.0
        )
        self.axes[2].set_title("Base model similarity")
        self.axes[2].axis("off")
        self.base_cbar = self.fig.colorbar(self.base_heatmap_overlay, ax=self.axes[2], fraction=0.046, pad=0.04)
        self.base_cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)

        # Connect the mouse click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.tight_layout()
        print("UI ready! Click on the left image.")
        plt.show()

    def _extract_patch_embeddings(self, model, mean, std):
        """Run the image through a model and return L2-normalized patch
        embeddings arranged on the feature grid (h_feat, w_feat, hidden_dim)."""
        transform = T.Compose([
            T.Resize(self.target_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        # Generate the exact tensor shape the model expects (1, 3, H, W)
        pixel_values = transform(self.raw_image).unsqueeze(0).to(self.device)

        _, _, self.H, self.W = pixel_values.shape
        self.h_feat = self.H // self.patch_size
        self.w_feat = self.W // self.patch_size
        num_patches = self.h_feat * self.w_feat

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        last_hidden_state = outputs.last_hidden_state
        patch_embs = last_hidden_state[0, -num_patches:]  # (num_patches, hidden_dim)

        # L2 Normalize (dot product of normalized vectors = cosine similarity)
        patch_embs = F.normalize(patch_embs, p=2, dim=-1)
        return patch_embs.view(self.h_feat, self.w_feat, -1)  # (h_feat, w_feat, hidden_dim)

    @staticmethod
    def _similarity_map(patch_embs, feat_x, feat_y, orig_H, orig_W):
        """Cosine-similarity map of the selected patch vs all patches,
        upsampled to the original image resolution."""
        query_emb = patch_embs[feat_y, feat_x].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        sim_map = (patch_embs * query_emb).sum(dim=-1)  # (h_feat, w_feat)

        sim_tensor = sim_map.unsqueeze(0).unsqueeze(0)  # (1, 1, h_feat, w_feat)
        sim_upsampled = F.interpolate(
            sim_tensor,
            size=(orig_H, orig_W),
            mode="bilinear",
            align_corners=False
        )[0, 0].cpu().numpy()
        return sim_upsampled

    def on_click(self, event):
        # Ignore clicks outside the left image
        if event.inaxes != self.axes[0]:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Map pixel click coordinate to feature grid coordinate
        feat_x = int((x / self.orig_W) * self.w_feat)
        feat_y = int((y / self.orig_H) * self.h_feat)

        # Bound check
        feat_x = max(0, min(feat_x, self.w_feat - 1))
        feat_y = max(0, min(feat_y, self.h_feat - 1))

        # 1. Compute similarity maps for BOTH models using the same selected patch
        sim_upsampled = self._similarity_map(
            self.patch_embs, feat_x, feat_y, self.orig_H, self.orig_W
        )
        base_sim_upsampled = self._similarity_map(
            self.base_patch_embs, feat_x, feat_y, self.orig_H, self.orig_W
        )

        # 2. Update the visual overlays
        self.heatmap_overlay.set_data(sim_upsampled)
        self.axes[1].set_title(f"Fine-tuned similarity to Patch ({feat_x}, {feat_y})")
        self.base_heatmap_overlay.set_data(base_sim_upsampled)
        self.axes[2].set_title(f"Base similarity to Patch ({feat_x}, {feat_y})")

        # 3. Draw a red box on the left image to show exactly what patch was selected
        [p.remove() for p in reversed(self.axes[0].patches)]

        patch_w = self.orig_W / self.w_feat
        patch_h = self.orig_H / self.h_feat
        rect = plt.Rectangle(
            (feat_x * patch_w, feat_y * patch_h),
            patch_w, patch_h,
            edgecolor='red', facecolor='none', lw=2
        )
        self.axes[0].add_patch(rect)

        # Redraw the UI
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # Ensure you are running this in an environment with a desktop/GUI (not headless)
    DinoInteractiveExplorer(
        image_path=r"E:\RAW Data\LiquidDetetion\Archive\Alpha2\frame_28_1771945917220.jpg",
    )