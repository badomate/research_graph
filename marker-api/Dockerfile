FROM python:3.11-slim

# Minimal system deps — just curl for the healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080

# Proxy server: forwards PDF conversion requests to the Datalab Marker API
# POST /marker  { "filepath": "<path visible inside the container>" }
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
