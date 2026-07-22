# Thin child of the shared training base image. Build the base once on the DGX
# Spark from the repository root:
#   docker build -f Dockerfile.base -t dgx-train-base:latest .
FROM dgx-train-base:latest

WORKDIR /app

# Project-specific extras. torch, torchvision, transformers, huggingface_hub,
# and numpy already come from the base image.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy this analysis tool.
COPY . .

# Run the script with any CLI flags passed to `docker run`.
ENTRYPOINT ["python", "patch_similarity_evolution.py"]
