---
id: MS-002
title: 'M7 eval gate'
type: milestone
status: active
created: '2026-11-24T00:00:00+00:00'
updated: '2026-11-24T00:00:00+00:00'
target_release: mvp
sequence: 2
owner: SarmadRazaq
depends_on: [MS-001]
---

# MS-002: M7 eval gate

## Outcome
The M7 criterion (a 0 there is an automatic F) is met with evidence: an Approved golden set of about 59 Golden items, retrieval and answer metrics, a calibrated LLM judge and at least two configurations compared, all reproducible with `make eval` and written to `EVAL.md`.

## Measuring success
- [ ] Draft Golden items reviewed: each set to approved (with reviewer), fixed or dropped
- [ ] About 49 more Golden items written per `PROJECT_BLUEPRINT.md` §10.1 (Punjab vs Sindh, superseded vs current, Urdu and Roman Urdu, 8 Unanswerable)
- [ ] `make eval` runs end to end and writes the `EVAL.md` tables
- [ ] At least two retrieval or generation configs compared
- [ ] 30 answers labelled by hand in `evals/human_labels.csv` for judge calibration

## Non-goals
- The full R1–R8 / G1–G3 grid (stretch)
- Golden items over private Documents (the eval always uses Scope public)

## Contributing work items
- ISSUE-001 — Flaky test: test_concurrent_duplicate_uploads

## Notes
Imported from `docs/HUMAN_TODO.md` ("Before the eval runs").

---

## Changelog
| Version | Date       | Change summary       |
|---------|------------|----------------------|
| 1.0     | 2026-11-24 | Initial draft        |
