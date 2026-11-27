---
id: MS-001
title: 'MVP end to end'
type: milestone
status: active
created: '2026-11-24T00:00:00+00:00'
updated: '2026-11-24T00:00:00+00:00'
target_release: mvp
sequence: 1
owner: SarmadRazaq
---

# MS-001: MVP end to end

## Outcome
Every mandatory rubric line has a working path: the Official corpus is collected, ingested and indexed, `/ask` gives cited Answers in English, Urdu and Roman Urdu and refuses Unanswerable questions, and the frontend demo flow (upload, list, delete, chat, chunks panel) works end to end. Source: `PROJECT_BLUEPRINT.md` §16.1.

## Measuring success
- [ ] NVIDIA API key in `.env` and `scripts/check_models.py --probe` shows no MISSING model ids
- [ ] Tesseract with Urdu installed; OCR measured on ID-06 and 1–2 scanned Urdu Sources
- [ ] Blocked pages SP-04, SP-05, SP-06 saved; manual NADRA files moved to `corpus/raw/manual/`
- [ ] `scripts/seed_corpus.py` indexes every `core` and `pair` Source
- [ ] `/ask` returns cited Answers in en, ur and roman_ur, and a Refusal for an Unanswerable question
- [ ] A second User cannot see the first User's private Documents
- [ ] The demo flow works in the frontend against the real backend

## Non-goals
- Stretch work from §16.1: PPTX, quotas and fair GPU queue, Langfuse, deployment
- Verifying referral contacts (MS-003)

## Contributing work items
- ISSUE-002 — GO-9/10/11 notices: wire the emergency Notice (rubric M9 requires emergency referral)

## Notes
Imported from `PROJECT_BLUEPRINT.md` §16.1 and `docs/HUMAN_TODO.md` ("Today, before ingestion").

---

## Changelog
| Version | Date       | Change summary       |
|---------|------------|----------------------|
| 1.0     | 2026-11-24 | Initial draft        |
