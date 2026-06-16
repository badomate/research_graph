# Build from the repo root so the web app can ship the orchestrator's
# modules.store / modules.metadata alongside it:
#   docker build -f webapp/Dockerfile -t paper-webapp .
FROM python:3.11-slim

WORKDIR /app

COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

# Pipeline modules (Store + metadata) — webapp/main.py adds ./orchestrator to sys.path.
COPY orchestrator/ orchestrator/
COPY webapp/ webapp/

ENV DATABASE_URL=sqlite:////data/app.db \
    UPLOADS_DIR=/data/uploads \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data/uploads
EXPOSE 8000

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
