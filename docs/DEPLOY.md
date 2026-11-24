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
NVIDIA_API_KEY=nvapi-...
JWT_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(48))">   # >= 32 chars
CORS_ORIGINS=https://rag.example.org      # no *
LOG_QUESTION_TEXT=false
COOKIE_SECURE=1
FORWARDED_ALLOW_IPS=127.0.0.1             # address(es) of your reverse proxy
```

Use `QDRANT_API_KEY` (and `QDRANT_URL`) only for Qdrant Cloud; compose sets `QDRANT_URL` for the bundled Qdrant.

## 2. Check the models

```bash
python scripts/check_models.py --probe
```

Confirms the key and that each configured model id exists on the NVIDIA API.

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
- `GET /health/ready`: Qdrant and the NVIDIA API are reachable; 503 otherwise. Ollama is optional.

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

## Rotating secrets

- `JWT_SECRET`: change it in `.env` and `docker compose --profile api up -d api`. All sessions are invalidated.
- `NVIDIA_API_KEY`, `QDRANT_API_KEY`: update `.env`, recreate `api` the same way, then revoke the old key.
