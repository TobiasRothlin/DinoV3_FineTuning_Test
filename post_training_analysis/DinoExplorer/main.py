import io
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from transformers import AutoModel
from PIL import Image

app = FastAPI(title="DINOv3 Backend")

# Model Configuration from your source[cite: 1]
MEAN = [0.4114, 0.4183, 0.4359]
STD = [0.3005, 0.2981, 0.2966]
TARGET_SIZE = (2304, 832)
PATCH_SIZE = 16

# NOTE: Update this to your local model path or HuggingFace ID[cite: 1]
MODEL_PATH = r"C:\Users\TobiasRothlin\Downloads\dinov3-vitl16-finetuned-v1-SyntoGo-49\dinov3-vitl16-finetuned-v1-SyntoGo-49"

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Transformation pipeline based on your source[cite: 1]
transform = T.Compose([
    T.Resize(TARGET_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])


def encode_to_base64(tensor: torch.Tensor) -> str:
    """Converts a PyTorch tensor to a base64 encoded string of Float32 bytes."""
    np_array = tensor.cpu().numpy().astype(np.float32)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')


def encode_ndarray_to_base64(np_array: np.ndarray) -> str:
    """Converts a NumPy array to a base64 encoded string of Float32 bytes."""
    np_array = np.ascontiguousarray(np_array.astype(np.float32))
    return base64.b64encode(np_array.tobytes()).decode('utf-8')


def compute_pca_coords(patch_embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
    """PCA via SVD on raw patch embeddings. Returns (num_patches, n_components) float32."""
    centered = patch_embeddings - patch_embeddings.mean(axis=0, keepdims=True)
    # Vt rows are the principal axes (right singular vectors), ordered by variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return centered @ components.T


@app.get("/")
def serve_frontend():
    """Serves the Index.html file on the root call."""
    return FileResponse("index.html")


@app.post("/api/process-image")
async def process_image(file: UploadFile = File(...)):
    """Processes an uploaded image and returns embeddings and dimensions."""
    # 1. Read and prepare the image
    image_bytes = await file.read()
    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = raw_image.size

    # 2. Transform and run through model
    pixel_values = transform(raw_image).unsqueeze(0).to(device)
    _, _, H, W = pixel_values.shape

    h_feat = H // PATCH_SIZE
    w_feat = W // PATCH_SIZE
    num_patches = h_feat * w_feat

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state

    # 3. Extract CLS and Patch Embeddings
    # DINOv3 standard format: [batch, 0] is the CLS token
    cls_token = last_hidden_state[0, 0]

    # Extract patches based on your logic[cite: 1]
    patch_embs = last_hidden_state[0, -num_patches:]

    # PCA(3) on the RAW patch embeddings (before normalization), matching image_patch_pca.py
    raw_patch_np = patch_embs.cpu().numpy().astype(np.float32)
    pca_coords = compute_pca_coords(raw_patch_np, n_components=3)

    # L2 Normalize the patch embeddings as in your script[cite: 1]
    patch_embs = F.normalize(patch_embs, p=2, dim=-1)

    # 4. Return as Base64 for rapid network transfer
    return {
        "patch_size": PATCH_SIZE,
        "grid_shape": {"h_feat": h_feat, "w_feat": w_feat},
        "original_size": {"width": orig_w, "height": orig_h},
        "cls_vector_b64": encode_to_base64(cls_token),
        "patch_embeddings_b64": encode_to_base64(patch_embs),
        "pca_coords_b64": encode_ndarray_to_base64(pca_coords),
        "embedding_dim": patch_embs.shape[-1]
    }


if __name__ == "__main__":
    print("Starting DINOv3 Backend on http://127.0.0.1:8081")
    uvicorn.run(app, host="127.0.0.1", port=8081)