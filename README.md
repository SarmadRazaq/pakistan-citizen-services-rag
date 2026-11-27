# Citizen Services RAG (Pakistan)

A multilingual (**English · Urdu · Roman Urdu**) question-answering service for Pakistani citizen services and rights:
- ID cards
- passports
- the Sehat card
- individual tax basics
- complaints and rights
- Punjab and Sindh consumer protection and driving licences

Every answer comes only from official documents, with a citation and the document's date. The service gives **information, not legal advice**.

Built with **FastAPI**, a **LangGraph** answer pipeline, **Qdrant** hybrid search (dense + sparse, RRF, reranking), hosted OpenAI models for chat and embeddings plus a local GPU reranker (see DESIGN.md §4.7), and a **Next.js** frontend. Upload a document in the web app and the whole pipeline runs: validate → extract (OCR for scans) → clean → chunk → embed → store. Ask a question and you get a cited answer, the retrieved chunks with their scores, or a clear refusal.

## Documents in this repo

| File | What it is |
|---|---|
| [DESIGN.md](DESIGN.md) | Architecture, flows, and the justification of every design decision |
| [EVAL.md](EVAL.md) | Golden set, metrics, configuration comparison, failure analysis |
| [docs/REQUIREMENTS_CHECKLIST.md](docs/REQUIREMENTS_CHECKLIST.md) | Every requirement of the brief and rubric, with where it is met |
| [GLOSSARY.md](GLOSSARY.md), [docs/adr/](docs/adr/) | Domain terms and hard-to-reverse decisions |
| [notebooks/](notebooks/) | Step-by-step pipeline notebooks: extraction, chunking, embedding and retrieval, answering |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Production deployment with Docker Compose |
| `PROJECT_BLUEPRINT.md` | The original design blueprint, with the edge-case catalogue (§9) |
| `corpus/SEED_SOURCES.md` | Verified official sources and decisions D1–D5 |
| `corpus/MANUAL_COLLECTION.md` | Checklist for sources that block scripts (NADRA, PM Health, KP Sehat) |
| `docs/HUMAN_TODO.md` | Steps only the student can do (API key, Tesseract, golden set, referrals) |

## Setup (Windows, PowerShell)

Prerequisites:
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker Desktop (for Qdrant)
- Node 20+ (frontend)
- **Tesseract with Urdu and English language data**, for OCR of scanned pages
- an **OpenAI API key** with credit (answer models and embeddings), and a GPU for the local reranker (or `RERANKER_PROVIDER=none` without one): see `.env.example` and DESIGN.md §4.7

### 1. Backend

```powershell
.\scripts\dev.ps1 install      # venv + pinned dependencies
copy .env.example .env         # then set JWT_SECRET and OPENAI_API_KEY (never commit .env)
.\scripts\dev.ps1 qdrant       # Qdrant in Docker, bound to 127.0.0.1:6333
.\.venv\Scripts\python scripts\check_models.py --probe   # confirms your key and every model id
.\scripts\dev.ps1 run          # API on http://127.0.0.1:8000  (Swagger UI at /docs)
```

`GET /health/ready` reports each dependency and returns **503** until Qdrant is up and the configured API keys work.

### 2. Admin account and the official corpus

```powershell
.\.venv\Scripts\python scripts\create_admin.py admin@example.org      # asks for the password
.\.venv\Scripts\python scripts\collect_corpus.py                      # downloads the scriptable sources
.\.venv\Scripts\python scripts\collect_corpus.py --register-manual    # registers files saved by hand
.\.venv\Scripts\python scripts\seed_corpus.py                         # ingests them as the public corpus
```

Admins can also upload official documents in the web app (the "Official" switch on the upload panel).

### 3. Frontend

```powershell
cd frontend
copy .env.example .env.local   # API_BASE_URL=http://127.0.0.1:8000
npm install
npm run dev                    # http://localhost:3000
```

`NEXT_PUBLIC_USE_MOCKS=1` runs the UI with no backend, from built-in mock data.

### 4. Tests

```powershell
.\scripts\dev.ps1 check                              # ruff + mypy --strict + pytest
cd frontend; npm test; npm run typecheck; cd ..      # frontend unit tests and types
.\.venv\Scripts\python scripts\run_notebooks.py      # re-executes the pipeline notebooks
```

### 5. Evaluation

One command runs the whole evaluation and saves the results under `evals/results/<timestamp>/`:

```powershell
.\.venv\Scripts\python -m evals.run --split test
```

See [EVAL.md](EVAL.md) "How to reproduce" for the eval account, the configurations it compares and the offline re-scoring mode.

On Linux/macOS the same targets exist in the `Makefile` (`make install`, `make check`, `make eval`, …).

## Repository rules

- **Raw corpus files are never committed** (decision D5). Only the manifest, the seed list and the collection script are.
- Secrets live in `.env`, which is gitignored. `gitleaks` runs in CI.
- `qdrant_client` may be imported **only** in `app/retrieval/vector_store.py`. This is enforced by ruff, a test and CI, because tenant isolation depends on it (blueprint §5.8.2).
- `requirements.txt` is fully pinned. Add libraries to `requirements.in` and run `.\scripts\dev.ps1 pin`.

## Privacy: where your text goes

- **The official corpus** contains public government documents. Questions over it are answered by hosted models on the **OpenAI API**, which processes the prompt under its API terms.
- **Private uploads** follow `PRIVATE_CONTEXT_POLICY`:
  - `hosted_with_notice` (default): answers that use your uploaded text are generated through the hosted API, and the answer carries a "processed by a third-party model" notice. The upload screen says so before you upload.
  - `local_only`: answers that use your uploads run on the local Ollama model, so nothing private leaves the machine. Quality is lower, and those answers are marked *degraded*.
- Embeddings run on the hosted API too, so uploaded chunk text is sent there at upload. Reranking runs on the local GPU and Tesseract OCR locally. Each is a setting (`EMBEDDING_PROVIDER`, `RERANKER_PROVIDER`, `INJECTION_PROVIDER`); `EMBEDDING_PROVIDER=local` keeps chunk text on the machine.
- Logs never contain the question text unless `LOG_QUESTION_TEXT=true`. Secrets and CNIC-shaped numbers are redacted.

## Data sources and licensing

The corpus is built from **public documents published by Pakistani government bodies**, used for non-commercial coursework. **Raw files are not redistributed in this repository** (decision D5). The repository holds only the manifest (URL, retrieval date and SHA-256 of each file) and the script that rebuilds the corpus.

| Source | Used for | Terms |
|---|---|---|
| Federal Board of Revenue (fbr.gov.pk) | tax due dates, Active Taxpayer List explainers, Income Tax Ordinance, rate cards, Urdu Income Tax Rules | FBR Terms of Use allow unmodified copies for free, non-commercial information sharing |
| NADRA (nadra.gov.pk) | CNIC/CRC/FRC procedures, fee structure, Registration Policy (EN/UR) | No reuse licence; saved manually for local use only, cited, never redistributed |
| Directorate General of Immigration & Passports (dgip.gov.pk) | passport requirements, FAQ, Passport Rules 1974/2021, fee notices | All IP rights reserved; local use only, cited |
| PM National Health Programme / KP Sehat Card Plus | Sehat card steps, FAQs, coverage | No terms published; local use only, cited |
| Pakistan Code / Senate of Pakistan | Constitution (English) | Pakistan Code: "for general informational purposes"; refer to the Gazette for the authentic text |
| Wafaqi Mohtasib, PM Delivery Unit, Ministry of Human Rights | complaint procedures, Citizen Portal manual, helpline | No reuse licence; local use only, cited |
| Punjab Code, Punjab Consumer Protection Council, DLIMS, CTP Lahore | Punjab consumer law, guidelines, driving licence | No reuse licence; local use only, cited |
| Sindh Laws, Sindh Judicial Academy, Sindh Police DLS | Sindh consumer law, licence fees | "All rights reserved"; local use only, cited |
| BISP | **excluded** (its disclaimer forbids copying or conversion); permission requested | — |

Every answer names the source document, page and date, and links to the official URL. **The service gives general information, not legal or tax advice.** Personal-case questions are referred to the relevant office.
