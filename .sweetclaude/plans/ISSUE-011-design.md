# ISSUE-011 design: background ingestion

Goal: make the failing tests in `tests/integration/test_background_ingestion.py` and the new tests in
`tests/api/test_ingestion_api.py` pass. The sync (201) path keeps its current behaviour exactly.

## Config (`app/config.py`)
- Add `sync_ingest_max_pages: int = 80`, `background_ingest_workers: int = 2` (ge=1),
  `background_ingest_queue_max: int = 20` (ge=1). `sync_ingest_max_bytes` already exists.
- Add the three to `.env.example` with one-line comments.

## Errors (`app/core/errors.py`)
- `INGEST_IN_PROGRESS = (409, "Ingestion in progress")` in the 409 group.

## Page count (`app/ingestion/validation.py`)
- `pdf_page_count(path: Path) -> int` (pymupdf, `stream=path.read_bytes()` as `check_pdf` does). Other types → no page trigger.

## Service (`app/ingestion/service.py`)
State on `IngestionService.__init__`:
- `self._tasks: dict[str, asyncio.Task[None]]` (doc_id → running job; strong refs),
- `self._workers = asyncio.Semaphore(settings.background_ingest_workers)`.

Helpers:
- `_is_large(file, file_type) -> bool`: `size_bytes > settings.sync_ingest_max_bytes`, or a PDF with
  `pdf_page_count > settings.sync_ingest_max_pages` (run in a thread). Return the page count too, for `pages_total`.
- Split `_job_slot` into `_acquire_slot(ctx, owner)` / `_release_slot(owner)`; keep `_job_slot` as the
  context manager built on them. The background path acquires inline and releases in the job's `finally`.
- Queue bound: before accepting a background job, `len(self._tasks) >= background_ingest_queue_max` →
  `AppError(TOO_MANY_INGEST_JOBS, ..., headers={"Retry-After": "10"})`.
- `_ensure_idle(doc_id)`: if `doc_id in self._tasks` → `AppError(INGEST_IN_PROGRESS, ...)`. Call it in
  `patch` and `replace` (after the 404/403 checks, so foreign ids still 404).
- `_spawn(doc_id, coro)`: `asyncio.create_task`, store in `_tasks`, `add_done_callback` that pops it and
  logs `ingest.job_crashed` if the task ended with an unexpected exception (jobs handle their own errors).
- `drain()`: `await asyncio.gather(*self._tasks.values(), return_exceptions=True)` in a loop until empty.
- `aclose()`: cancel every task, then gather with `return_exceptions=True`.

Upload:
- Fast inline part (both paths): owner, public-topic check, file name, `_check_quota`, slot, `_validate`,
  `_check_file_duplicate`. Then `_is_large`. Sync path: continue exactly as today inside the slot.
- Background path: queue check; create the row (PENDING, request metadata, title = request title or
  `_title_from_file_name`, `text_sha256=None`, `pages_total` = PDF page count or 0, `pages_done=0`); the
  existing IntegrityError → DUPLICATE_DOCUMENT handling applies. Copy the file into
  `_doc_dir(owner, doc_id) / f"original.{file_type}"` (thread) BEFORE returning: the router deletes the temp
  file. Spawn `_run_upload_job(...)`, return `(self._out(row), True)`. Release the slot inline only if
  something fails before the spawn.
- Job: `async with self._workers, self._lock(doc_id)`: status PROCESSING → `_extract` on the stored copy →
  update `pages_total` from `extracted.page_count` → `_check_content_duplicate(..., file_sha=...)` →
  `_check_chunk_quota` → `_infer_meta` → update the row's inferred fields + text_sha256/language/pages/is_ocr
  → the shared index step → READY. `except asyncio.CancelledError`: shielded `_fail(ctx, doc_id, "Ingestion
  was interrupted.")`, but skip the row update if the row is gone or DELETING; re-raise.
  `except Exception`: `_fail`. `finally`: `_release_slot(owner)`.
- Factor the shared index step (plan → embed/upsert with progress → verify → store manifest/file → READY
  update) into one helper used by sync and background, so the code is not duplicated. Progress: after each
  embed batch, `pages_done = round(pages_total * embedded / len(planned))` (monotonic), written with
  `_update`; at READY, `pages_done = pages_total`. The sync path may skip intermediate writes.
- `_fail` must accept a message as well as an exception (or wrap the message in an AppError).
- Background jobs run with the uploader's `TenantContext`; keep that ctx in the closure.

Replace:
- Lock → load → writable → not DELETING → `_ensure_idle` → slot → validate → NO_CHANGE → file duplicate →
  `_is_large`. Sync: as today.
- Background: queue check; copy the file to `_doc_dir / f"incoming.{file_type}"`; record `previous_status`;
  set UPDATING, `pages_done=0`, `pages_total=page count`; spawn `_run_replace_job`; return `(row, True)`.
- Job: `async with self._workers, self._lock(doc_id)`: extract incoming → content duplicate (exclude doc) →
  chunk quota → the diff-upsert body (factored out of `replace`, shared with sync) with progress →
  `_store_files` from the incoming copy → delete `incoming.*` → READY. Failure or cancel:
  `_compensate_update(...)` (shielded on cancel) and delete `incoming.*`. `finally`: release slot.

Delete (E28):
- First, if `doc_id in self._tasks`: `task.cancel()` and `await asyncio.gather(task, return_exceptions=True)`.
  Then the existing delete runs unchanged (it removes points, files and row).

Lock note: the request releases the doc lock before the job takes it. `_tasks` registration happens
before the lock is released, so `_ensure_idle` closes that gap.

## Wiring (`app/main.py`)
- Append the `IngestionService` to `closables` so `_close` calls `aclose()` at shutdown.
- `app/protocols.py` `DocumentService`: add `drain()` and `aclose()`; update the docstring. Update any
  fake DocumentService in tests if the Protocol change breaks type checks.

## Docs
- Service module docstring: describe sync vs background.
- `PROJECT_BLUEPRINT.md` §9 rows E5, E6, E28: name the new tests; drop "not implemented (ISSUE-011)".
- `docs/DEPLOY.md`: note that background jobs are in-process (one uvicorn worker), and a crash mid-job
  leaves PROCESSING rows until the ISSUE-012 reconciler exists.
- `frontend/src/lib/api-types.ts` / `openapi.json`: add `ingest_in_progress` where error codes are listed
  (patch the stored JSON minimally; do not regenerate it).

## Done when
`.venv/Scripts/python.exe -m pytest -q` all green, `ruff check .`, `ruff format --check .`, `mypy app` clean.
