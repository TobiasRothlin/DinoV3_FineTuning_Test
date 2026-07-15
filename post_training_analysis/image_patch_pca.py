import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA


def visualize_dinov3_gpu(
        image_path,
        model_id= r"C:\Users\TobiasRothlin\Downloads\dinov3-vitl16-finetuned-v1-SyntoGo\dinov3-vitl16-finetuned-v1-SyntoGo",
        target_size=(448, 448),
        overlay_alpha=0.6
):
    # 1. Setup device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    # Load model directly onto your target device
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # 2. Load original image and save its dimensions
    patch_size = 16
    raw_image = Image.open(image_path).convert("RGB")
    orig_W, orig_H = raw_image.size  # Keep original dimensions for plotting

    # Calculate target dimensions that are multiples of patch_size
    target_h = (target_size[0] // patch_size) * patch_size
    target_w = (target_size[1] // patch_size) * patch_size

    # Create the specific resized image for the model
    model_input_image = raw_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Process inputs and move the resulting tensors to the GPU
    inputs = processor(images=model_input_image, return_tensors="pt", do_resize=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    _, _, H, W = inputs["pixel_values"].shape
    h_feat, w_feat = H // patch_size, W // patch_size
    num_patches = h_feat * w_feat

    print(f"Original Resolution: {orig_H}x{orig_W}")
    print(f"Model Input Resolution: {H}x{W}")
    print(f"Feature Map Grid: {h_feat}x{w_feat} ({num_patches} patches)")

    # 3. Pass image through DINOv3 (Inference on GPU)
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Extract patch embeddings
    last_hidden_state = outputs.last_hidden_state

    # Extract only the spatial patches, discarding CLS and register tokens
    patch_embeddings_tensor = last_hidden_state[0, -num_patches:]

    # Move to CPU for scikit-learn's PCA (which runs on CPU)
    patch_embeddings = patch_embeddings_tensor.cpu().numpy()

    # 5. Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_embeddings)

    # 6. Min-Max normalize to [0, 1] for RGB rendering
    pca_min = pca_features.min(axis=0)
    pca_max = pca_features.max(axis=0)
    pca_rgb = (pca_features - pca_min) / (pca_max - pca_min)
    pca_rgb_grid = pca_rgb.reshape(h_feat, w_feat, 3)

    # 7. Move back to GPU for fast Bilinear Interpolation
    pca_rgb_tensor = torch.tensor(pca_rgb_grid, device=device).permute(2, 0, 1).unsqueeze(0)

    # Upsample to the ORIGINAL image dimensions instead of the model input dimensions
    pca_rgb_upsampled_tensor = F.interpolate(
        pca_rgb_tensor,
        size=(orig_H, orig_W),
        mode="bilinear",
        align_corners=False
    )[0].permute(1, 2, 0)

    # Bring back down to CPU numpy array for visualization plotting
    pca_rgb_upsampled = pca_rgb_upsampled_tensor.cpu().numpy()

    # 8. Create the alpha blend overlay using the ORIGINAL image
    img_float = np.array(raw_image).astype(float) / 255.0
    overlay_img = (1.0 - overlay_alpha) * img_float + overlay_alpha * pca_rgb_upsampled

    # 9. Plotting the 4 side-by-side visualizations
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Plot 1: Original Image
    axes[0].imshow(raw_image)
    axes[0].set_title(f"Original ({orig_W}x{orig_H})")
    axes[0].axis("off")

    # Plot 2: Patch Visualization (scaled to original dimensions)
    axes[1].imshow(raw_image)
    axes[1].set_title(f"Patch Grid (Scaled to {orig_W}x{orig_H})")
    axes[1].axis("off")

    # Calculate the effective patch size on the original resolution
    effective_patch_w = orig_W / w_feat
    effective_patch_h = orig_H / h_feat

    # Draw scaled grid lines
    for x in range(w_feat + 1):
        axes[1].axvline(x * effective_patch_w - 0.5, color='white', linewidth=0.5, alpha=0.5)
    for y in range(h_feat + 1):
        axes[1].axhline(y * effective_patch_h - 0.5, color='white', linewidth=0.5, alpha=0.5)

    # Plot 3: PCA Embeddings Only (upsampled to original)
    axes[2].imshow(pca_rgb_upsampled)
    axes[2].set_title("PCA Patch Embeddings")
    axes[2].axis("off")

    # Plot 4: PCA Overlayed on Original
    axes[3].imshow(overlay_img)
    axes[3].set_title("PCA Embeddings Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example: Running it at a specific model resolution, but outputting at original image resolution
    visualize_dinov3_gpu(
        r"E:\RAW Data\LiquidDetetion\Archive\Alpha2\frame_97_1771945928221.jpg",
        target_size=(2304, 832)
    )