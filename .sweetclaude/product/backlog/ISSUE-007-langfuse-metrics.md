---
id: ISSUE-007
title: 'Langfuse tracing and /metrics'
type: net-new-feature
status: new
created: '2026-11-24T00:00:00+00:00'
updated: '2026-11-24T00:00:00+00:00'
priority: P3
---

# ISSUE-007: Langfuse tracing and /metrics

**Priority:** P3
**Depends on:** ISSUE-006
**GitHub:** https://github.com/SarmadRazaq/pakistan-citizen-services-rag/issues/8

## Summary
Stretch item 5: Langfuse traces for /ask and a /metrics endpoint.

## Initial Thinking
- `trace_id` already exists on AskResponse; wire it to Langfuse.
- `request_log` already feeds /metrics/summary.
