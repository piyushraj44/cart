FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ backend/
COPY ml/ ml/
COPY data/processed/ data/processed/
COPY data/samples/ data/samples/
COPY --from=frontend /build/frontend/dist frontend/dist
# Keep cold starts independent of model-registry availability. If a local build
# context already includes the encoder, this simply loads the existing files.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('/app/ml/artifacts/minilm' if __import__('pathlib').Path('/app/ml/artifacts/minilm/config.json').exists() else 'sentence-transformers/all-MiniLM-L6-v2').save('/app/ml/artifacts/minilm')"
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.hosted:app --host 0.0.0.0 --port ${PORT:-8000}"]
