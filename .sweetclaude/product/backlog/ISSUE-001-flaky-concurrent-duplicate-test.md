---
id: ISSUE-001
title: 'Flaky test: test_concurrent_duplicate_uploads'
type: bug-fix
status: new
created: '2026-11-24T00:00:00+00:00'
updated: '2026-11-24T00:00:00+00:00'
priority: P2
milestone: MS-002
---

# ISSUE-001: Flaky test: test_concurrent_duplicate_uploads

**Priority:** P2
**Depends on:** —
**GitHub:** https://github.com/SarmadRazaq/pakistan-citizen-services-rag/issues/2

## Summary
`tests/integration/test_ingestion_service.py::test_concurrent_duplicate_uploads` fails about 1 run in 10 (2/10 with current code, 1/10 on the code before 2026-11-24's changes). It asserts exactly one of two concurrent identical uploads gets DUPLICATE_DOCUMENT; sometimes neither or both do.

## Initial Thinking
- Likely a race between the file-hash duplicate check and the insert: both uploads pass `_check_file_duplicate` before either row commits, and the unique constraint on (user_id, file_sha256) surfaces as a different error or a purge of a FAILED row.
- Check how `IntegrityError` from `uq_documents_owner_file` is mapped (service.py around the second `_check_file_duplicate` call).
- A flaky test in CI erodes trust in the suite before the eval gate (MS-002).

## Open Questions
- Is the bug in the service (real race) or only in the test's timing?
