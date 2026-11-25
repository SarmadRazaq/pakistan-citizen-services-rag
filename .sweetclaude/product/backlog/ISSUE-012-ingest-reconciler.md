---
id: ISSUE-012
title: 'Startup reconciler for crashed ingestion (E29, E41)'
type: net-new-feature
status: new
created: '2026-11-25T00:00:00+00:00'
updated: '2026-11-25T00:00:00+00:00'
priority: P2
---

# ISSUE-012: Startup reconciler for crashed ingestion (E29, E41)

**Priority:** P2
**Depends on:** —
**GitHub:** https://github.com/SarmadRazaq/pakistan-citizen-services-rag/issues/13

## Summary
If the API process dies mid-ingest, Documents stay PROCESSING/UPDATING/DELETING with partial vectors or files. The blueprint's reconciler (E29) and the partial-failure retry for account deletion (E41) do not exist; only in-process error handling cleans up today.

## Initial Thinking
- On startup, find Documents stuck in a non-terminal status older than a grace period; remove their points and files and mark FAILED (or finish a DELETING).
- Re-run interrupted account deletions until count == 0.
- Idempotent, logged, covered by a test that simulates a crash between steps.
