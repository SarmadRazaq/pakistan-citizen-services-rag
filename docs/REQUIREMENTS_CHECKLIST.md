# Requirements checklist

Every requirement in the brief (`Portfolio-01-Production-RAG-Service.pdf`) and the grading rubric
(`Portfolio-01-Grading-Rubric-RAG.xlsx`), with where it is met and what is still open.
Checked against the code on 25 Nov 2026.

**Status:** ✅ done · 🟡 partly done · ❌ missing · 👤 needs the student (cannot be done by code alone)

## Grade-critical gaps (do these first)

| Gap | Rubric | Marks at risk | Owner |
|---|---|---|---|
| Measured evaluation: golden set has 10 of 25+ items, no eval run yet | M7 | 12, **and a 0 here means an F** | 👤 key + corpus, then run `evals/run.py` |
| Configuration comparison, judge check and failure analysis with real numbers in `EVAL.md` | M8 | 8 | after M7 |
| Service must honour `X-Eval-Config` (or the comparison runs one config per restart) | M8 | part of 8 | code |
| `GET /metrics/summary` returns 501; no cost tracking; no dashboard | B3 | up to 4 bonus | code |
| README is stale ("scaffold", "embeddings run locally") | M14 | part of 3 | docs |
| `DESIGN.md` | M3, M4, M11 | 9 + 4 + part of 6 | being written |
| Executed Jupyter notebooks | M11 | part of 6 | being written |
| Public repo, features merged into `main`, pushed | M13 | 4 | 👤 approve the push |
| Swagger screenshots, AI-usage docs | M14 | 3 | 👤 screenshots; AI docs can be drafted |
| Demo video covering all 14 items | M15 | 8, and any feature not shown can lose its marks | 👤 |

## Mandatory

### Scenario: upload → ingest → retrieve → answer

| Requirement | Status | Where |
|---|---|---|
| User uploads documents; the upload starts the whole pipeline | ✅ | `POST /api/v1/documents` → [app/ingestion/service.py](../app/ingestion/service.py): validate → extract (OCR) → clean → chunk → embed → store → READY |
| Ingest: extract, chunk, embed, store | ✅ | `app/ingestion/`, `app/retrieval/hosted.py`, Qdrant |
| Retrieve the most relevant passages | ✅ | hybrid dense + sparse, RRF, rerank, relevance gate (`app/retrieval/`) |
| Answer grounded with citations, or refuse | ✅ | LangGraph pipeline [app/generation/graph.py](../app/generation/graph.py) |
| Code-based, no auto-builder (Flowise, Langflow, Dify) | ✅ | own code + LangGraph |

### Section 2: technical requirements

| Requirement | Rubric | Status | Where |
|---|---|---|---|
| Layered structure: config, schemas, ingestion, retrieval, generation, guardrails, main, evals, tests | M1 | ✅ | `app/config.py`, `app/schemas.py`, `app/ingestion/`, `app/retrieval/`, `app/generation/`, `app/guardrails/`, `app/main.py`, `evals/`, `tests/` |
| No business logic in routes | M1 | ✅ | `app/api/*.py` only call services |
| Secrets from environment only | M1, M12 | ✅ | pydantic-settings + `.env` (gitignored), `.env.example` |
| Frontend: upload, list documents, ask, read answers with citations | M2 | ✅ | Next.js: `frontend/src/components/documents/`, `frontend/src/components/chat/` |
| Each answer shows retrieved chunks, **chunk count** and **similarity score** | M2 | ✅ | [ChunksPanel.tsx](../frontend/src/components/chat/ChunksPanel.tsx) |
| `DESIGN.md` justifying: corpus, text extraction, chunking, embedding model, vector DB, retrieval settings, LLM, prompt design (choice, why, alternatives, trade-offs) | M3 | 🟡 being written | content exists in `RAG_DESIGN_PLAN.md` and `PROJECT_BLUEPRINT.md` |
| Decisions backed by evaluation numbers | M3 | ❌ | needs the eval run |

### Section 3: required features

| Requirement | Rubric | Status | Where |
|---|---|---|---|
| **A. API** of your own design, explained in DESIGN.md | M4 | ✅ API / 🟡 doc | `/auth/signup`, `/auth/login`, `/auth/me`; `/documents` (POST, GET, GET `{id}`, PATCH, PUT, DELETE, GET `{id}/chunks`); `/ask`, `/search`, `/feedback`; `/metrics/summary`; `/health/live`, `/health/ready` |
| **B.** PDF and DOCX | M5 | ✅ | `app/ingestion/extractors/` |
| **B.** Metadata on every chunk: doc id, file name, page/section, position | M5 | ✅ | `ChunkPayload` in `app/protocols.py` (doc_id, file_name, page_start/end, section_path, chunk_index, char_start/end) |
| **B.** Duplicate uploads detected (file hash) | M5 | ✅ | file hash, text hash, and against the public corpus → 409 (`test_duplicate_upload`, `test_content_duplicate`) |
| **B.** Delete removes vectors; deleted content never in answers | M5 | ✅ | `test_deleted_doc_not_retrievable`, `test_delete_removes_vectors_files_and_row` |
| **C.** Answers based on retrieved context | M6 | ✅ | prompt rules + relevance gate |
| **C.** Every citation points to a retrieved chunk | M6 | ✅ | citation verification in the graph; unknown ids are dropped |
| **C.** Says clearly when documents lack the answer; never invents | M6 | ✅ | Refusal outcome (`outcome: REFUSAL`) |
| **D1.** Golden set ≥ 25, labelled source + reference answer, ≥ 5 unanswerable, reworded and multi-part | M7 | ❌ 10 items | [evals/golden_set.jsonl](../evals/golden_set.jsonl), schema in `evals/golden.py` |
| **D2.** Retrieval metrics: hit rate@k, recall@k, MRR | M7 | 🟡 code ✅, no results | `evals/metrics.py` |
| **D3.** Answer metrics: faithfulness, correctness, unanswerable handling | M7 | 🟡 code ✅, no results | `evals/judge.py`, `evals/ragas_eval.py` |
| **D4.** Frameworks (RAGAS, LLM judge, Langfuse…) justified in EVAL.md; judge reliability checked | M8 | 🟡 code ✅ (`evals/calibration.py`), no results | `EVAL.md` sections exist |
| **D5.** ≥ 2 configurations compared, winner picked by numbers | M8 | 🟡 configs exist, but the service **ignores the `X-Eval-Config` header** the eval sends, so R1/R5/G2 cannot be measured yet; no results | `evals/configs/`: R1 dense only, R4 hybrid + rerank, R5 hybrid no rerank, CB closed book, G2 other generator |
| **D6.** One command runs the whole eval and saves results; EVAL.md with failure analysis | M8 | 🟡 command ✅, no results | `python -m evals.run` (see `EVAL.md` "How to reproduce") |
| **E.** Input guardrails: empty, too long, off-topic, prompt injection, jailbreak | M9 | ✅ | `app/guardrails/input_guards.py` (multilingual injection and jailbreak patterns, decoded variants, optional injected classifier) |
| **E.** Output guardrails: no invented facts/citations, no system prompt or key leaks | M9 | ✅ | `app/guardrails/output_guards.py` |
| **E.** Each guardrail proven in the video | M9, M15 | 👤 | video |
| **F.** Error codes 200, 201, 204, 400, 404, 409, 415, 422, 429, 503, consistent format | M10 | ✅ | RFC 9457 problem+json, `app/core/errors.py` |
| **G.** OpenAPI docs at `/docs` with descriptions and examples | M4, M14 | ✅ | FastAPI; screenshots 👤 |

### Section 5: testing

| Requirement | Rubric | Status | Where |
|---|---|---|---|
| Notebooks first: extraction, chunking, embedding, retrieval, answering | M11 | 🟡 being written | `notebooks/` |
| Then API-based tests | M11 | ✅ | `tests/api/` |
| Success and expected-failure cases, executed | M11 | ✅ | about 690 pytest tests passing + frontend `node --test` |
| Approach justified in DESIGN.md | M11 | 🟡 being written | |
| Notebooks submitted | M11 | 🟡 | |

### Section 6: Git and GitHub

| Requirement | Rubric | Status | Where |
|---|---|---|---|
| Public repository | M13 | ❌ repo is private | 👤 make public |
| `main` is the stable branch, features merged into it | M13 | ❌ `main` has only the first commit; work is on `feature/mvp-integration` | 👤 approve merge + push |
| Feature branches | M13 | ✅ | `feature/*`, `issue-*` branches |
| Regular meaningful commits | M13 | ✅ | |
| No API keys committed; `.env` in `.gitignore`; `.env.example` | M13 | ✅ | |

### Section 7: code quality

| Requirement | Rubric | Status |
|---|---|---|
| Comments, docstrings, type hints, clear names | M12 | ✅ (mypy and ruff clean) |
| Settings in one module, secrets in env | M12 | ✅ |
| Timeouts and retries on every external call (LLM, embeddings, vector DB) | M12 | ✅ |
| Async I/O; parallel where it helps | M12 | ✅ |
| Each request logged with id, latency, errors; no secrets logged | M12 | ✅ structlog, `X-Request-ID` |
| Exceptions handled; no duplicated code | M12 | ✅ |

### Sections 8–9: AI usage and submission

| Requirement | Rubric | Status |
|---|---|---|
| 1. Link to public GitHub repo | M13 | ❌ private |
| 2. Clean source code | M1 | ✅ |
| 3. Jupyter notebooks | M11 | 🟡 being written |
| 4. `requirements.txt` pinned, `.env.example` | M14 | ✅ |
| 5. README: setup, config, run service, tests, eval | M14 | 🟡 stale: still says "Status: scaffold" and that embeddings run locally |
| 6. `DESIGN.md` | M3 | 🟡 being written |
| 7. `EVAL.md` with golden set, metrics, comparison, failure analysis | M7, M8 | 🟡 structure only |
| 8. Swagger UI screenshots | M14 | ❌ `docs/screenshots/` empty, 👤 |
| 9. AI usage: prompts, chat history, tools and models, configuration | M14 | ❌ 👤 export chat history; doc can be drafted |
| Demo video, 14 items | M15 | 👤 |

## Bonus

| Bonus | Rubric | Status | Where / gap |
|---|---|---|---|
| A. Signup, login, JWT, hashed passwords, own documents only | B1 (5) | ✅ | `app/api/auth.py`, `app/core/tenancy.py`, `test_cross_tenant_access_is_404_and_public_writes_403` |
| B. Multilingual: Urdu / Roman Urdu questions, answer in the question's language, separate results | B2 (4) | 🟡 works ✅, separate results ❌ | needs multilingual golden items + eval run |
| C. Tracing per step, p50/p95, tokens, cost per question, error rate, dashboard | B3 (4) | ✅ | per-step latency in the answer's debug drawer and the `ask.done` log line; `GET /metrics/summary` ([app/observability/metrics.py](../app/observability/metrics.py)) gives p50/p95, tokens, cost and list price per question, error and fallback rate; dashboard at `/metrics`. Hosted calls count as billed (`LLM_BILLED=false` zeroes the cost); list price needs `MODEL_PRICES`. Langfuse is configured but never called |
| D. Modern frontend | B4 (3) | ✅ | Next.js + Tailwind |
| E. Extra file types (TXT, MD, PPTX…) | B5 (2) | ✅ | PPTX, HTML, TXT, MD |
| F. Hidden instructions treated as data; corrupt/malicious files rejected | B6 (3) | ✅ | `test_doc_injection_not_followed`, `test_zip_bomb`, `test_macro_documents`, `test_corrupt_files` |
| G. Edge cases (8+ handled **and shown**) | B7 (4) | ✅ handled, 👤 show | `PROJECT_BLUEPRINT.md` §9 catalogue names a test per case |
| H. Deployment with public URL in README | B8 (3) | 🟡 | Docker + compose + `docs/DEPLOY.md`; not deployed |

### Edge cases from the brief

| Edge case | Status |
|---|---|
| Scanned or empty PDF | ✅ OCR, or 400 `no_extractable_text` / `empty_file` |
| Same file uploaded twice | ✅ 409 |
| Very large file / very long document | ✅ 413 above the cap; 🟡 202 background ingestion being added on this branch (ISSUE-011) |
| Question before any upload | ✅ |
| Question the documents cannot answer | ✅ Refusal |
| Question about a deleted document | ✅ |
| Two documents that contradict each other | ✅ prompt rule (newer / official wins, both cited); golden items still to write |
| Tables, special characters, non-English text | ✅ |
| LLM or embedding API timing out / rate-limiting | ✅ retries, model fallback chain, 503 |
| Several uploads or questions at the same time | ✅ `test_concurrent_duplicate_uploads`, `test_concurrent_asks` |

## Demo video checklist (brief p. 8)

| Item | Ready to show? |
|---|---|
| Intro: corpus, counts, file types | needs the corpus ingested |
| Architecture walkthrough: structure, upload flow, question flow | `DESIGN.md` diagrams |
| Design decisions and why | `DESIGN.md` |
| Upload a PDF and a DOCX in the frontend; list updates | ✅ |
| 3+ cited answers with the chunks panel (count + scores) | needs the corpus and the key |
| Unanswerable question | ✅ |
| Delete, then the content no longer appears | ✅ |
| Guardrails incl. prompt injection and jailbreak | ✅ |
| 2+ error cases (unsupported type 415, duplicate 409) | ✅ |
| Each edge case | ✅ see the table above |
| Run the eval, walk through EVAL.md | needs the eval run |
| Run the tests | ✅ `pytest -q` |
| Swagger UI | ✅ |
| Each bonus feature | see the bonus table |
