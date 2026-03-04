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

# Run on CPU; override at runtime with -e TORCH_DEVICE=cuda if you have a GPU
ENV TORCH_DEVICE=cpu

EXPOSE 8080

# marker_server is the CLI entry-point installed by marker-pdf
# POST /marker  { "filepath": "<path visible inside the container>" }
CMD ["marker_server", "--port", "8080"]
