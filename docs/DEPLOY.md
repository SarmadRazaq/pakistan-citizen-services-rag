# Deploying the service

One API process per container, one Qdrant, one web frontend, on a single host. All models are hosted
(NVIDIA API); only Tesseract OCR runs locally, so no GPU is needed.

## Prerequisites

- Docker with Compose v2.
- An NVIDIA API key (build.nvidia.com, starts with `nvapi-`).
- A reverse proxy for TLS (Caddy, nginx, Traefik) on the host.

## 1. Configure

```bash
cp .env.example .env
```

Set at least these in `.env` (the API refuses to start in `prod` otherwise, and lists every problem):

```
APP_ENV=prod
LLM_API_KEY=nvapi-...                   # NVIDIA default; for OpenAI see "Without an NVIDIA key" below
JWT_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(48))">   # >= 32 chars
CORS_ORIGINS=https://rag.example.org      # no *
LOG_QUESTION_TEXT=false
COOKIE_SECURE=1
FORWARDED_ALLOW_IPS=127.0.0.1             # address(es) of your reverse proxy
```

Leave `EVAL_OVERRIDES_ENABLED` unset (false) in production: it makes `/ask` and `/search` honour the
`X-Eval-Config` header (dense-only, no rerank, another generation model) for admin callers, and exists only
for the evaluation harness.

Use `QDRANT_API_KEY` (and `QDRANT_URL`) only for Qdrant Cloud; compose sets `QDRANT_URL` for the bundled Qdrant.

## 2. Check the models

```bash
python scripts/check_models.py --probe
```

Confirms the key and that each configured model id exists on the configured endpoint (NVIDIA or OpenAI), and probes exactly the retrieval providers selected; skipped ones are listed.

## 3. Start

```bash
docker compose --profile api up -d --build
```

This starts `qdrant`, `api` (127.0.0.1:8000) and `web` (127.0.0.1:3000). The API image bakes the BGE-M3
tokenizer, so no Hugging Face download happens at runtime.

## 4. Create the admin

Interactive: the password is prompted, never passed as an argument.

```bash
docker compose --profile api exec api python scripts/create_admin.py admin@example.org
```

Add `-it` (`docker compose --profile api exec -it api ...`) if your Compose version does not allocate a TTY by
default. Use `--demote` to remove the role.

## 5. Seed the Official corpus

After `scripts/collect_corpus.py` has produced `corpus/raw` and `corpus/manifest.csv`:

```bash
.venv/Scripts/python scripts/seed_corpus.py            # Windows; use python on Linux
```

It is idempotent: re-running reports unchanged, replaced and new documents. It writes through the normal ingestion
path to the same database and Qdrant, so run it where the data stores are reachable (point `QDRANT_URL` at
`http://127.0.0.1:6333`, and `DATABASE_URL` at the SQLite file in the `api_data` volume or run it inside the
container with the corpus mounted).

## Health

- `GET /health/live`: process is up (used by the container healthcheck).
- `GET /health/ready`: Qdrant and the LLM endpoint (`llm_api`) are reachable, plus `openai_embeddings` when `EMBEDDING_PROVIDER=openai`; 503 otherwise. Ollama is optional.

## Without an NVIDIA key (OpenAI)

NVIDIA's API catalog is not available in every country. Everything can run on an OpenAI key instead: chat
(`gpt-4.1-mini`, `gpt-4o-mini`), the judge (`gpt-4o`) and embeddings (`text-embedding-3-large`, requested at
1024 dimensions so the Qdrant collection keeps its size). The reranker runs locally (`RERANKER_PROVIDER=local`,
needs a GPU) or is switched off (`none`: retrieval gates on dense similarity, as in the `rerank=false` eval
override; re-tune `TAU_KEEP` / `TAU_REFUSE`). The NemoGuard jailbreak classifier is NVIDIA-only, so set
`INJECTION_PROVIDER=none` (the heuristic input guards still run).

```
LLM_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...        # used for chat and embeddings while LLM_BASE_URL is OpenAI's
EMBEDDING_PROVIDER=openai
RERANKER_PROVIDER=none
INJECTION_PROVIDER=none
```

The full block (models, prices, rate limit) is in `.env.example` under "OpenAI setup". `LLM_BASE_URL` and
`LLM_API_KEY` are the neutral names of the old `NVIDIA_BASE_URL` / `NVIDIA_API_KEY`; both spellings work.
`APP_ENV=prod` then requires the LLM key and, with `EMBEDDING_PROVIDER=openai`, an OpenAI key, and asks for an
NVIDIA key only if an NVIDIA retrieval provider is still selected. OpenAI usage is billed, so `cost_usd` counts
it (set `MODEL_PRICES`). Switching embedding provider changes the vectors: use a new collection and re-seed.

## TLS

Terminate TLS in a reverse proxy on the host and forward to `127.0.0.1:3000` (web) and, if the API is public,
`127.0.0.1:8000`. Send `X-Forwarded-For` and `X-Forwarded-Proto`, and list the proxy address in
`FORWARDED_ALLOW_IPS`, or rate limits will see the proxy as the only client. In `prod` the API adds
`Strict-Transport-Security`; keep `COOKIE_SECURE=1`.

## Backups

State lives in two Docker volumes: `api_data` (SQLite database and uploaded files) and `qdrant_storage` (vectors).
Stop the stack (or snapshot the volumes) and archive both, for example:

```bash
docker compose --profile api stop api web
docker run --rm -v <project>_api_data:/v -v "$PWD":/b alpine tar czf /b/api_data.tgz -C /v .
docker run --rm -v <project>_qdrant_storage:/v -v "$PWD":/b alpine tar czf /b/qdrant_storage.tgz -C /v .
docker compose --profile api start api web
```

The Qdrant index can be rebuilt by re-ingesting, but the SQLite database and uploads cannot.

## Scaling

Run exactly one API process per container (the Dockerfile pins `--workers 1`). Document locks, ingestion jobs and
rate limits are in process memory. Do not raise workers or replicas of `api` without moving that state out first.

Large uploads (above `SYNC_INGEST_MAX_MB` or `SYNC_INGEST_MAX_PAGES`) are ingested by background jobs inside the API
process (`BACKGROUND_INGEST_WORKERS` at a time, `BACKGROUND_INGEST_QUEUE_MAX` in flight). A graceful stop marks running
jobs FAILED ("interrupted"). A crash or kill mid-job leaves the document PROCESSING (or UPDATING) until the ISSUE-012
reconciler exists; delete and re-upload it.

## Rotating secrets

- `JWT_SECRET`: change it in `.env` and `docker compose --profile api up -d api`. All sessions are invalidated.
- `NVIDIA_API_KEY`, `QDRANT_API_KEY`: update `.env`, recreate `api` the same way, then revoke the old key.
