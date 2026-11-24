---
id: ISSUE-002
title: 'GO-9/10/11 notices: wire the emergency Notice'
type: net-new-feature
status: done
created: '2026-11-24T00:00:00+00:00'
updated: '2026-11-24T00:00:00+00:00'
priority: P2
milestone: MS-001
---

# ISSUE-002: GO-9/10/11 notices: wire the emergency Notice

**Priority:** P2
**Depends on:** —
**GitHub:** https://github.com/SarmadRazaq/pakistan-citizen-services-rag/issues/3

## Summary
Stretch item 4 in PROJECT_BLUEPRINT.md §16.1. `EMERGENCY_GUARD=true` exists in config and the frontend renders an emergency strip, but no code path ever produces `NoticeCode.EMERGENCY`, so an emergency question gets no emergency Referral.

## Initial Thinking
- Detect emergencies in the question (pattern list en/ur/roman-ur plus a query-understanding flag), per blueprint GO-11.
- Attach Referrals whose `triggers` include `emergency` (referrals.yaml already has police 15 and rescue 1122, unverified).
- Emergency Referrals are shown first, then any relevant Answer.
- Verify GO-9 (not advice) and GO-10 (dated information) are complete while here.

## Open Questions
- Should an emergency question still be answered from the corpus, or only get the Notice?
