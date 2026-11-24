---
id: ISSUE-003
title: 'Enforce the closed Topic list for public Documents'
type: tech-debt
status: new
created: '2026-11-24T00:00:00+00:00'
updated: '2026-11-24T00:00:00+00:00'
priority: P3
---

# ISSUE-003: Enforce the closed Topic list for public Documents

**Priority:** P3
**Depends on:** —
**GitHub:** https://github.com/SarmadRazaq/pakistan-citizen-services-rag/issues/4

## Summary
GLOSSARY.md makes Topic a closed list for public Documents, but `topic` is free text up to 60 characters at upload and metadata edit. A public Document tagged with an unknown topic never matches a Referral.

## Initial Thinking
- Validate `topic` against `app.ingestion.metadata.TOPICS` when visibility is public (upload and PATCH).
- Keep private Documents free-text and optional.
- Expose the list in the API schema (enum) so the frontend can offer a dropdown.

## Open Questions
- Reject unknown topics with 422, or coerce to null with a warning?
