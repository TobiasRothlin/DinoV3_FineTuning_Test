# Thin child of the universal training base. Build the base once on the Spark:
#   docker build -f Dockerfile.base -t dgx-train-base:latest .
FROM dgx-train-base:latest

# Working directory for this project's sources.
WORKDIR /app

# Project-specific extras only (HF_HOME + CUDA alloc config come from the base).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training sources (main.py and the sibling modules).
COPY . .

# Persist teacher checkpoints (CHECKPOINT_DIR="checkpoints"). The shared
# /hf_cache volume is inherited from the base image.
VOLUME ["/app/checkpoints"]

# Run the training script.
ENTRYPOINT ["python", "main.py"]
