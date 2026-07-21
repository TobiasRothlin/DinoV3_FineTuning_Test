# DinoV3_FineTuning_Test

## Universal training container (DGX Spark)

The Docker setup is split into a reusable **base image** and thin **per-project
child images**, so every training project shares the same CUDA/PyTorch stack.

### 1. Build the base once (on the Spark)

```bash
docker build -f Dockerfile.base -t dgx-train-base:latest .
```

- `Dockerfile.base` starts from `nvcr.io/nvidia/pytorch:25.12-py3` (multi-arch,
  resolves to arm64/sbsa on the GB10 hardware).
- Installs the shared libs from `requirements-base.txt` (numpy, tqdm,
  transformers, huggingface_hub, mlflow, torchinfo). `torch` comes from NGC.
- Sets `HF_HOME=/hf_cache` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Rebuild the base only when you bump the CUDA/PyTorch tag or shared deps.

### 2. Build & run this project

Each project has its own `Dockerfile` (`FROM dgx-train-base:latest`) that adds
only project-specific extras from `requirements.txt` and runs `main.py`.

With docker compose (recommended — encodes GPU flags and mounts):

```bash
docker compose up --build
```

Or manually:

```bash
docker build -t dinov3-finetune .
docker run --gpus all \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v dgx_hf_cache:/hf_cache \
  -v /home/spark/Datasets/AGI:/data:ro \
  --shm-size 16g \
  dinov3-finetune
```

### Reusing for other projects

Copy `Dockerfile` + `docker-compose.yml` into the new project, adjust its
`requirements.txt` extras and the entrypoint command. The shared
`dgx_hf_cache` volume dedupes model downloads across all projects.


### Setup for Inference
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
pip install -r requirements-base.txt
```
