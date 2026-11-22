# MVP work packages (rev 5)

The MVP from `PROJECT_BLUEPRINT.md` §16.1 is built as **parallel work packages on separate branches**, cut from `feature/foundations`. Each package **owns** a set of files. It may read anything, but it edits only what it owns. The main session integrates the branches and wires `app/main.py`.

## Frozen contracts (read, never edit without the main session)

| File | What it fixes |
|---|---|
| `app/schemas.py` | API request/response shapes |
| `app/protocols.py` | `Extractor`, `Embedder`, `VectorStore`, `Reranker`, `Retriever`, `LLMClient`, `InjectionClassifier`, `DocumentService`, `AskService` + data types |
| `app/core/errors.py` | `ErrorCode` registry, `AppError`, problem+json |
| `app/core/tenancy.py` | `TenantContext` (`for_user`, `with_scope`, `ensure_readable/writable`, `write_owner_for`) |
| `app/core/ids.py` | id helpers, `sha256_text`, `sha256_file`, deterministic `point_id` |
| `app/db/models.py`, `app/db/session.py` | `User`, `Document`, `RequestLog`, `Feedback`; `Database.session(ctx)` with the ORM tenant filter |
| `app/config.py` | every setting. **Need a new one?** Add a module-level default in your own file and list it in your final report. |
| `app/api/deps.py` | `get_db`, `get_document_service`, `get_ask_service`, `get_tenant` (the auth package replaces only `get_tenant`'s body) |
| `tests/fakes.py` | `FakeEmbedder`, `InMemoryVectorStore`, `FakeReranker`, `FakeLLM`, `make_payload` |
| `requirements.in` / `.txt` | shared, pinned runtime libraries (already include PyMuPDF, trafilatura, python-docx, langchain-openai, langgraph, rapidfuzz, pyyaml, tokenizers) |

## Ownership

| Package | Branch | Owns | Delivers |
|---|---|---|---|
| **ING** ingestion | `feature/ingestion` | `app/ingestion/**`, `app/api/documents.py`, `scripts/collect_corpus.py`, `scripts/seed_corpus.py`, `tests/**/test_ingestion*.py`, `tests/fixtures/files/**` | validation, PDF/HTML/DOCX/TXT/MD extractors (+OCR fallback), cleaning, text-quality check (E46), language detection, chunking, `DocumentService` implementation (dedupe, upload, list, get, patch, chunks, diff-upsert `replace`, delete with 0-points check), corpus scripts |
| **RET** retrieval | `feature/retrieval` | `app/retrieval/**`, `app/core/device.py`, `requirements-ml.in`, `tests/**/test_retrieval*.py` | BGE-M3 embedder (lazy, GPU/CPU), `TenantScopedVectorStore` (the **only** `qdrant_client` import), bge-reranker, `Retriever` (variants → hybrid → RRF → dedupe → rerank → scores) |
| **AUTH** auth + LLM | `feature/auth-llm` | `app/core/security.py`, `app/core/resilience.py`, `app/api/auth.py`, the body of `get_tenant` in `app/api/deps.py`, `app/generation/llm.py`, `scripts/create_admin.py`, `tests/**/test_auth*.py`, `tests/**/test_llm*.py` | argon2 + JWT (`ver`), signup/login/me/delete-me, `get_tenant`, `LLMRouter` (NVIDIA chain → Ollama, timeouts, circuit breakers, rate limiter, JSON mode, eval disk cache) |
| **ASK** pipeline + guardrails | `feature/ask` | `app/generation/{prompts,query_understanding,answer,graph,service}.py`, `app/guardrails/{input_guards,output_guards,patterns/**}`, `app/api/ask.py`, `tests/**/test_ask*.py`, `tests/**/test_guard*.py` | LangGraph pipeline, `AskService` (ask/search/feedback), JSON answer parsing + repair, quote verification, relevance gate, basic input/output guards |
| **EVAL** evaluation | `feature/evals` | `evals/**` (except `golden_set.jsonl` content), `requirements-eval.in`, `tests/**/test_eval*.py` | metrics (Hit@k, Recall@k, MRR, nDCG, refusal accuracy), LLM judge, runner with configs + response cache, report → EVAL.md tables, `make eval` |
| **UI** frontend | `feature/frontend` | `frontend/**` | upload/list/delete, chat with citations, chunks panel (fetched/used/cited + scores), auth pages, notices, RTL Urdu |

## Rules for every package

1. **Branch:** create your branch from `feature/foundations`, and commit there. **Never push**; the main session reviews and pushes. Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
2. **Environment:** `UV_LINK_MODE=copy uv venv --python 3.12 .venv`, then `UV_LINK_MODE=copy uv pip install --python .venv -r requirements.txt -r requirements-dev.txt`.
3. **Checks:** `.venv/Scripts/python -m ruff check app tests scripts`, `ruff format --check`, `.venv/Scripts/python -m mypy`, `.venv/Scripts/python -m pytest`. All must pass before you finish.
4. **Tests without a network:** use `tests/fakes.py`. Never call the NVIDIA API, download model weights or need Qdrant/Docker in default tests. Anything live goes under `@pytest.mark.live`.
5. **Off-limits:** don't edit `app/main.py`, frozen contracts or another package's files. If you need a contract change, stop and say exactly what and why in your report.
6. **Tenant isolation:** every data access takes the `TenantContext`. Another user's private resource → 404 (`ctx.ensure_readable`). Public-corpus writes by non-admins → 403.
7. **Secrets:** read keys only from `Settings`. Never log them or the question text (unless `log_question_text`).
8. **Final report:** what you built, files touched, new settings needed, tests added and their result, and anything left undone.
