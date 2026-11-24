# Supersession is decided at answer time, not stored

When two public Documents on the same subject (same Agency, Jurisdiction and Topic) are both cited, the answer treats the one with the later Effective date as current and the other as superseded. Nothing records which Document supersedes which: the prompt gives each Citation its agency, jurisdiction and date, and the model compares them. We chose this because the seed corpus has only a handful of superseded-vs-current pairs, Effective dates are already captured at ingestion, and an explicit link would need manual curation for every pair and still wouldn't cover user uploads.

## Consequences

- The superseded-vs-current items in the golden set test the prompt and the model's date reading, not the data model. A regression there means "fix the prompt", not "fix the link".
- Documents with no Effective date are never current or superseded relative to each other, and private Documents never take part in Supersession (see `GLOSSARY.md`).
- If the eval shows the model regularly misjudging which Document is current, the fallback is to add a stored `supersedes` link for public Documents. Choosing that later means a schema change and back-filling every pair by hand.
