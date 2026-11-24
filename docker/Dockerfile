# API image. Every model is hosted (NVIDIA API); only Tesseract OCR runs locally, so no GPU is needed.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \n    HF_HOME=/srv/.hf

# Tesseract with Urdu + English data is needed for OCR (scanned PDFs, images, junk text layers).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-urd tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

USER app

# Bake the BGE-M3 tokenizer (tokenizer.json only, no weights; app/ingestion/chunking.py loads it with
# Tokenizer.from_pretrained(settings.embedding_model) to size Chunks) so runtime needs no Hugging Face
# download. The build fails if it cannot be fetched.
RUN python -c "from tokenizers import Tokenizer; from app.config import Settings; Tokenizer.from_pretrained(Settings(_env_file=None).embedding_model)"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live').status==200 else 1)"

# Exactly ONE uvicorn worker: document locks, ingestion jobs and rate limits live in process memory,
# so more than one worker per container would break them. Scale by running separate deployments with
# a shared-state redesign, not by raising --workers.
# --proxy-headers only trusts X-Forwarded-* from FORWARDED_ALLOW_IPS (the reverse proxy's address).
CMD ["sh", "-c", "exec uvicorn app.main:app_factory --factory --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-127.0.0.1}\""]
