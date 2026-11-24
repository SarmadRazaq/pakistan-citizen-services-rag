---
id: ISSUE-006
title: 'Multi-tenant quotas and fair GPU queue'
type: net-new-feature
status: new
created: '2026-11-24T00:00:00+00:00'
updated: '2026-11-24T00:00:00+00:00'
priority: P3
---

# ISSUE-006: Multi-tenant quotas and fair GPU queue

**Priority:** P3
**Depends on:** ISSUE-005
**GitHub:** https://github.com/SarmadRazaq/pakistan-citizen-services-rag/issues/7

## Summary
Stretch item 3: per-User quotas and a fair queue for GPU embedding/reranking work.

## Initial Thinking
- Chunk quota exists (`max_chunks_per_user`); add request/upload quotas and queue fairness.
