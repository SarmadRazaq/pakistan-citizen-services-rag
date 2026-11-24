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
| `docs/HUMAN_TODO.md` | The prerequisites only the student can do (key, Tesseract, golden set, referrals) |
| `DESIGN.md`, `EVAL.md`, `AI_USAGE.md` | Written at the end from measured results |

## Setup (Windows, PowerShell)

Prerequisites:
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker Desktop
- Node 20+ (frontend)
- **Tesseract with Urdu and English language data**, for OCR
- an **NVIDIA API key** (build.nvidia.com), for LLM answers

```powershell
.\scripts\dev.ps1 install      # venv + pinned dependencies
copy .env.example .env         # then set JWT_SECRET and NVIDIA_API_KEY (never commit .env)
.\scripts\dev.ps1 qdrant       # Qdrant in Docker, bound to 127.0.0.1:6333
.\.venv\Scripts\python scripts\check_models.py --probe   # confirms your key and the model ids
.\scripts\dev.ps1 run          # API on http://127.0.0.1:8000  (docs at /docs)
.\scripts\dev.ps1 check        # ruff + mypy --strict + pytest
```

On Linux/macOS the same targets exist in the `Makefile` (`make install`, `make check`, …).

Production deployment: see [docs/DEPLOY.md](docs/DEPLOY.md).

`GET /health/ready` reports each dependency. It returns **503** until Qdrant is up and the NVIDIA key works. Ollama is optional unless `OLLAMA_REQUIRED=true`.

## Repository rules

- **Raw corpus files are never committed** (decision D5). Only the manifest, the seed list and the collection script are.
- Secrets live in `.env`, which is gitignored. `gitleaks` runs in CI.
- `qdrant_client` may be imported **only** in `app/retrieval/vector_store.py`. This is enforced by ruff, a test and CI, because tenant isolation depends on it (blueprint §5.8.2).
- `requirements.txt` is fully pinned. Add libraries to `requirements.in` and run `.\scripts\dev.ps1 pin`.

## Privacy: where your text goes

- **The official corpus** contains public government documents. Questions over it are answered by open-weight models hosted on the **NVIDIA API**, which processes the prompt under NVIDIA's API terms.
- **Private uploads** follow `PRIVATE_CONTEXT_POLICY`:
  - `hosted_with_notice` (default): answers that use your uploaded text are generated through the NVIDIA API, and the answer carries a "processed by a third-party model" notice. The upload screen says so before you upload.
  - `local_only`: answers that use your uploads run on the local Ollama model, so nothing private leaves the machine. Quality is lower, and those answers are marked *degraded*.
- Embeddings, reranking and the injection classifier always run **locally**.
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
