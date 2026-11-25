---
id: ISSUE-011
title: 'Background ingestion with 202 and progress (E5, E6)'
type: net-new-feature
status: new
created: '2026-11-25T00:00:00+00:00'
updated: '2026-11-25T00:00:00+00:00'
priority: P2
---

# ISSUE-011: Background ingestion with 202 and progress (E5, E6)

**Priority:** P2
**Depends on:** —
**GitHub:** https://github.com/SarmadRazaq/pakistan-citizen-services-rag/issues/12

## Summary
Uploads above SYNC_INGEST_MAX_MB should return 202 Accepted and ingest in the background, reporting pages_done/pages_total through GET /documents/{id}. Today IngestionService.upload and replace always ingest inline and return accepted=False, so the 202 route path and DocumentOut.progress are never used, and a 200-page document (the Constitution) holds the request open.

## Initial Thinking
- Spawn a supervised background task per accepted upload; update pages_done as batches embed.
- Keep the per-document lock and per-user job slots; delete during ingest must cancel the task (E28).
- A crashed task must leave the Document FAILED, not PROCESSING (ties to ISSUE-012).
- One API process per container, so an in-process task queue is enough.
