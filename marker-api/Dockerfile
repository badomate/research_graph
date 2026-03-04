FROM python:3.11-slim

# System libraries required by marker's vision/OCR dependencies
# (OpenCV, Pillow, surya all need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first — avoids pulling the massive CUDA wheel
# (~800 MB vs ~2.5 GB for the default CUDA build)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# marker-pdf and the lightweight FastAPI server it ships with
RUN pip install --no-cache-dir \
    marker-pdf \
    uvicorn \
    fastapi \
    python-multipart

# ── Runtime environment ───────────────────────────────────────────────────────
# Run on CPU; override at runtime with -e TORCH_DEVICE=cuda if you have a GPU
ENV TORCH_DEVICE=cpu

# All model weights are stored under /model_cache, which is mapped to the
# persistent `marker_models` named volume in docker-compose.yml.
# Setting all three variables ensures every library (HuggingFace, PyTorch hub,
# surya XDG cache) writes to the same location and survives container rebuilds.
ENV HF_HOME=/model_cache \
    TORCH_HOME=/model_cache/torch \
    XDG_CACHE_HOME=/model_cache/xdg

# Pre-create the directories so they are owned by the process user and the
# volume mount does not shadow a non-existent path.
RUN mkdir -p /model_cache/torch /model_cache/xdg

EXPOSE 8080

# marker_server is the CLI entry-point installed by marker-pdf
# POST /marker  { "filepath": "<path visible inside the container>" }
CMD ["marker_server","--host", "0.0.0.0", "--port", "8080"]
