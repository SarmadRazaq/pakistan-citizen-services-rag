# ISSUE-011 acceptance criteria: background ingestion (E5, E6, E28)

Blueprint §5.7: files up to the sync limit are ingested inline (201). Larger ones (up to the hard cap) are
accepted with **202 + `Location`** and ingested in the background. The client polls `GET /documents/{id}`
and reads `progress.pages_done / progress.pages_total`.

## Settings
- `sync_ingest_max_mb` (exists, default 10): files with `size_bytes > sync_ingest_max_mb * 1 MiB` go to
  the background. `0` therefore sends every non-empty file to the background (handy in tests).
- `sync_ingest_max_pages` (new, default 80): a PDF with more pages than this goes to the background, even
  if it is small.
- `background_ingest_workers` (new, default 2): at most this many background jobs run at once; the rest wait.
- `background_ingest_queue_max` (new, default 20): accepted-but-unfinished background jobs across all users.
  One more → 429 `too_many_ingest_jobs` with a `Retry-After` header, and nothing is stored.

## Upload (`IngestionService.upload`, `POST /documents`)
1. Small file: unchanged. Returns `accepted=False`, status READY (201 at the API).
2. Large file: returns `accepted=True` (202 at the API, `Location: /api/v1/documents/{id}`) with status
   PENDING or PROCESSING. After `await service.drain()` the document is READY, `chunk_count` > 0, its chunks
   are in the vector store, `chunks()` lists them, and `progress.pages_done == progress.pages_total`.
3. The caller deletes the uploaded temp file as soon as `upload` returns (the router does this). The
   background job must still succeed.
4. These checks still fail **inline**, before any 202, and leave no row: unsupported type (415), spoofed
   extension, empty file, file duplicate (409 `duplicate_document`), duplicate of the public corpus,
   document/storage quota (413), per-user job limit (429), non-admin public upload (403).
5. Request metadata (title, agency, jurisdiction, topic, effective_date) wins over inference, as for sync.
6. Progress: with a gated embedder and a small `embed_batch_size`, while the job is mid-way the document
   is PROCESSING and `0 <= pages_done <= pages_total`, `pages_total > 0` for a multi-page PDF; pages_done
   never decreases between polls and reaches pages_total at READY.
7. Background failure (the embedder raises on a batch): the document ends FAILED with a non-empty
   `error`, no points remain in the store, and the user's job slot is released (a later upload works).
   Uploading the same file again after FAILED succeeds (the failed row is purged).
8. Content duplicate found only after extraction (same text, different file bytes): the background job
   ends FAILED with an `error` that says the text already exists; no points are stored.
9. A background job counts against `max_ingest_jobs_per_user` until it finishes: with the limit at 1 and
   one job held by a gated embedder, a second upload by the same user is 429; after the job finishes, a
   new upload is accepted. The public (admin) corpus is exempt, as today.

## Replace (`PUT /documents/{id}`)
10. Small replacement: unchanged (`accepted=False`, READY, version + 1).
11. Large replacement: `accepted=True`, status UPDATING, version unchanged until the job ends. While it
    runs, the old version's points are still in the store. After `drain()`: READY, version + 1, new text
    served, old-only chunks removed.
12. Large replacement whose job fails: the document ends READY_STALE with `error` set, the old version's
    points are all still there, and the points this attempt added are gone.
13. NO_CHANGE (identical file) and file duplicate stay inline 409s.

## Jobs in progress
14. While a background job runs for a document, `patch` and `replace` on it fail with 409 and the code
    `ingest_in_progress` (new `ErrorCode.INGEST_IN_PROGRESS`, HTTP 409, title "Ingestion in progress").
15. E28: `delete` while a background job runs cancels the job and completes normally: the row, its points
    and its stored files are all gone, the job does not recreate anything afterwards, and the job slot is
    released.
16. `await service.aclose()` (called at app shutdown) cancels running jobs; each affected new upload ends
    FAILED (not PENDING/PROCESSING) with an error saying ingestion was interrupted, and leaves no points.
17. `await service.drain()` returns once every background job has finished (no-op when there are none).

## API (`tests/api/test_ingestion_api.py` style)
18. `POST /api/v1/documents` with a large file → 202, a `Location` header equal to
    `/api/v1/documents/{id}`, and a body with status PENDING or PROCESSING. After draining the service,
    `GET` on that Location → 200, status READY, `progress.pages_done == progress.pages_total`.
19. `PATCH` on a document whose job is running → 409 problem+json with `code: ingest_in_progress`.
