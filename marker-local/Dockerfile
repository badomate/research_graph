FROM python:3.11-slim

# ── Build-time switch ─────────────────────────────────────────────────────────
# CPU (default): TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# GPU (CUDA 12.1): TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# ── System libraries ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Step 1: install marker-pdf with default torch (gets deps right) ───────────
RUN pip install --no-cache-dir \
    marker-pdf \
    uvicorn \
    fastapi \
    python-multipart

# ── Step 2: reinstall torch/torchvision from the correct index ────────────────
# This overwrites whatever torch marker-pdf pulled in with the correct
# CPU or GPU wheel. Done after marker-pdf so all other deps are already
# resolved and pip won't touch torch again.
RUN pip install --no-cache-dir --force-reinstall \
    torch \
    torchvision \
    --index-url ${TORCH_INDEX_URL}

# ── Model cache ───────────────────────────────────────────────────────────────
ENV HF_HOME=/model_cache \
    TORCH_HOME=/model_cache/torch \
    XDG_CACHE_HOME=/model_cache/xdg

RUN mkdir -p /model_cache/torch /model_cache/xdg

# ── Runtime device ────────────────────────────────────────────────────────────
ENV TORCH_DEVICE=cpu

# ── Verify entry point exists at build time ───────────────────────────────────
RUN which marker_server

EXPOSE 8080

CMD ["marker_server", "--host", "0.0.0.0", "--port", "8080"]