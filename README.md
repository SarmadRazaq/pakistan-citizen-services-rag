# Citizen Services RAG (Pakistan)

A multilingual (**English · Urdu · Roman Urdu**) question-answering service for Pakistani citizen services and rights:
- ID cards
- passports
- the Sehat card
- individual tax basics
- complaints and rights
- Punjab and Sindh consumer protection and driving licences

Every answer comes only from official documents, with a citation and the document's date. The service gives **information, not legal advice**.

> **Status:** scaffold (work package P0). The API contract, error envelope, tenancy rules and health checks are in place. Ingestion, retrieval, answering, auth and evaluation land in work packages A0–A8 (see `PROJECT_BLUEPRINT.md` §15).

## Documents in this repo

| File | What it is |
|---|---|
| `PROJECT_BLUEPRINT.md` | The design: every decision, edge case and work package |
| `corpus/SEED_SOURCES.md` | Verified official sources and decisions D1–D5 |
| `corpus/MANUAL_COLLECTION.md` | Checklist for sources that block scripts (NADRA, PM Health, KP Sehat) |
| `DESIGN.md`, `EVAL.md`, `AI_USAGE.md` | Written at the end from measured results |

## Setup (Windows, PowerShell)

Prerequisites:
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker Desktop
- Node 20+ (frontend)
- **Tesseract with Urdu and English language data**, for OCR
- the AntSeed buyer proxy, for LLM answers

```powershell
.\scripts\dev.ps1 install      # venv + pinned dependencies
copy .env.example .env         # then set JWT_SECRET
.\scripts\dev.ps1 qdrant       # Qdrant in Docker, bound to 127.0.0.1:6333
antseed buyer start            # local OpenAI-compatible proxy on 127.0.0.1:8377
.\scripts\dev.ps1 run          # API on http://127.0.0.1:8000  (docs at /docs)
.\scripts\dev.ps1 check        # ruff + mypy --strict + pytest
```

On Linux/macOS the same targets exist in the `Makefile` (`make install`, `make check`, …).

`GET /health/ready` reports each dependency. It returns **503** until Qdrant and the AntSeed proxy are both reachable. Ollama is optional unless `OLLAMA_REQUIRED=true`.

## Repository rules

- **Raw corpus files are never committed** (decision D5). Only the manifest, the seed list and the collection script are.
- Secrets live in `.env`, which is gitignored. `gitleaks` runs in CI.
- `qdrant_client` may be imported **only** in `app/retrieval/vector_store.py`. This is enforced by ruff, a test and CI, because tenant isolation depends on it (blueprint §5.8.2).
- `requirements.txt` is fully pinned. Add libraries to `requirements.in` and run `.\scripts\dev.ps1 pin`.
