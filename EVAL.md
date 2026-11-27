# Evaluation

How well the RAG service retrieves, answers, cites and refuses, in English, Urdu and Roman Urdu.
The plan is `PROJECT_BLUEPRINT.md` §10. Every table between `<!-- EVAL:START … -->` and
`<!-- EVAL:END … -->` markers is **generated** by `python -m evals.run` (or re-rendered with
`python -m evals.report evals/results/<timestamp>`). Edit the prose around the markers, never inside.

## How to reproduce

```powershell
# 1. the service is running (uvicorn on :8000) and the official corpus is ingested
# 2. an eval account exists; its credentials come from the environment, never from git
$env:EVAL_EMAIL = "eval@example.com"; $env:EVAL_PASSWORD = "<the eval account password>"
# 3. OPENAI_API_KEY (or LLM_API_KEY) is in .env (judge + closed-book baseline)
# 4. the service runs with EVAL_OVERRIDES_ENABLED=true and the eval account is an ADMIN; otherwise the
#    X-Eval-Config header is ignored and R1/R5/G2 are reported as "NOT honoured"
.\.venv\Scripts\python -m evals.run --split test                 # approved items, default configs
.\.venv\Scripts\python -m evals.run --include-drafts --split all # while the golden set is in draft
.\.venv\Scripts\python -m evals.run --offline --split test       # re-score from the caches only
```

- Results: `evals/results/<timestamp>/{metrics.json, per_question.csv, answers.jsonl, report.md}`.
- Every service response (`evals/.cache/api/`), judge call (`evals/.cache/judge/`) and closed-book call
  (`evals/.cache/closed_book/`) is cached on disk, keyed by sha256 of the request. `--offline`
  reproduces the published numbers without the service or the LLM API; `--no-cache` forces fresh calls.
- `metrics.json` records the golden-set sha256, the judge model and prompt version, the generator
  models and the exact command.

<!-- EVAL:START run_info -->
_Not run yet._
<!-- EVAL:END run_info -->

## Golden set

`evals/golden_set.jsonl`, schema in `evals/golden.py` (blueprint §10.1). Gold labels are at
**document + page/section** level, so they stay valid across chunking configs. Only items with
`review_status: approved` are evaluated unless `--include-drafts` is passed; `dropped` items never run.

<!-- EVAL:START golden_set -->
_Not run yet._
<!-- EVAL:END golden_set -->

## Metrics

| Layer | Metric | Definition (`evals/metrics.py`) | Over |
|---|---|---|---|
| Retrieval | Hit@1/3/5/10 | ≥ 1 relevant chunk in the top k of `/search` | answerable |
| Retrieval | Recall@5/10 | share of the item's gold sources covered in the top k | answerable |
| Retrieval | MRR@10 | 1 / rank of the first relevant chunk | answerable |
| Retrieval | nDCG@10 | gain 2 = right file + page/section, 1 = right file only | answerable |
| Refusal | precision / recall / F1 | positive class = refuse (`answerable: false` or a refusal reason other than `partial`) | all |
| Refusal | false-refusal rate | answerable items refused | answerable |
| Answer | citation validity | answered items whose citations **all** point at gold-relevant chunks (no citation = invalid) | answered, answerable |
| Answer | notices | every expected notice (`not_advice`, `dated_information`, `referral`) present | items expecting notices |
| Multilingual | language match | detected answer language (`evals/language.py`) = expected | answered |
| Judge | correctness 0–2, citation support, refusal appropriateness, advice boundary | `evals/judge.py`, GPT-OSS-120B, temperature 0 | see rubric |
| Ops | p50 / p95 latency, tokens | `latency_ms.total` and `usage` from `/ask` | all |

A chunk is **relevant** when it comes from the gold source's Source (gold sources name a seed ID such as
`ID-05`, and corpus files are named `<ID>__slug.ext`, so the ID prefix must match exactly) **and** its page
span overlaps the gold pages, or, for HTML/DOCX/MD, its section path matches the gold section
(`evals/relevance.py`). Naming the Source, not the file, keeps labels valid when a Source is re-collected.

## Configuration comparison

Configs live in `evals/configs/*.yaml` (blueprint §10.6). Winner rule, fixed before running: maximize
**MRR@10 on the test split**, tie-break by Hit@5, then p95 latency.

| Config | What it changes | Needs service support? |
|---|---|---|
| `R4_hybrid_rerank` (**baseline**) | nothing: the service defaults (hybrid dense+sparse, RRF, rerank) | no |
| `R1_dense_only` | `X-Eval-Config: {"retrieval_mode":"dense","rerank":false}` | **yes** (admin + `EVAL_OVERRIDES_ENABLED`) |
| `R5_hybrid_no_rerank` (opt-in) | `X-Eval-Config: {"retrieval_mode":"hybrid","rerank":false}` | **yes** (admin + `EVAL_OVERRIDES_ENABLED`) |
| `G2_gpt4o_mini_generator` (opt-in) | `X-Eval-Config: {"generation_model":"gpt-4o-mini"}` (baseline generator: `gpt-4.1-mini`) | **yes** (admin + `EVAL_OVERRIDES_ENABLED`) |
| `CB_closed_book` | generator answers with no sources, called directly by the harness | no |

**Service-side support.** Overrides travel as a compact JSON `X-Eval-Config` request header on `/search`
and `/ask`. The service honours it only when `EVAL_OVERRIDES_ENABLED=true` (default false) and the caller
is an admin; for anyone else the header is ignored. Supported keys: `config` (label), `retrieval_mode`
(`dense` | `sparse` | `hybrid`), `rerank` (bool) and `generation_model` (must be in `GENERATION_MODELS`;
the generate call then uses only that model). Invalid values are a 422 `validation_failed`; applied
overrides are echoed in `X-Eval-Config-Applied`. When a service is not configured this way the harness
**falls back** to the defaults and marks the row "NOT honoured", so it is never mistaken for a real
measurement. The request params (`top_k`, `answer_language`, `scope`) always work.

**Relevance gate without the reranker.** `tau_keep` and `tau_refuse` are tuned for reranker scores. With
`rerank=false` the retrieved chunks keep their fused order, `rerank_score` is null, and both the context cut
(`tau_keep`) and the refusal gate (`tau_refuse`) use the best dense cosine similarity instead. Refusal and
context rates therefore differ from the default pipeline, which is part of what R1 and R5 measure.

<!-- EVAL:START config_comparison -->
_Not run yet._
<!-- EVAL:END config_comparison -->

### By question type

<!-- EVAL:START by_type -->
_Not run yet._
<!-- EVAL:END by_type -->

## Generator comparison

_Placeholder._ G1–G3 (`gpt-4.1-mini` as the baseline, `gpt-4o-mini`, and a third model if one is added) run on the winning retrieval
config once the service honours `generation_model` in `X-Eval-Config`. Selection rule, fixed in
advance: maximize judge correctness, subject to faithfulness ≥ 0.85, refusal accuracy ≥ 0.8 and
multilingual answer-language accuracy ≥ 0.9; tie-break by p95 latency, then fallback rate.

## Judge calibration

The judge is `gpt-4o`: a different, larger model than the generators (`gpt-4.1-mini`, `gpt-4o-mini`). It is the same vendor, so some self-preference bias remains possible, which is what the calibration below measures. To check it
(blueprint §10.5):

1. `python -m evals.calibration export --results evals/results/<timestamp>` writes a stratified sample
   of 30 baseline answers to `evals/human_labels.csv`, **without** the judge's scores.
2. Label every row: `human_correctness` 2 / 1 / 0, `human_citation_supported` y / n.
3. `python -m evals.calibration score --results evals/results/<timestamp>` reports % agreement and
   Cohen's κ (target ≥ 0.6) and refreshes the table below.

<!-- EVAL:START judge_calibration -->
_No human labels yet._
<!-- EVAL:END judge_calibration -->

## Multilingual

Urdu and Roman Urdu are reported separately from English, never averaged into one number.

<!-- EVAL:START multilingual -->
_Not run yet._
<!-- EVAL:END multilingual -->

## Contamination

The closed-book baseline asks the generator the same questions **without any sources**. A high
closed-book score means the model already knows the answer from pre-training; the RAG gain over it is
what retrieval contributes.

<!-- EVAL:START contamination -->
_Not run yet._
<!-- EVAL:END contamination -->

## Failure analysis

Each failed item is categorized automatically (`error`, `false_refusal`, `refusal_error`,
`retrieval_miss`, `ranking`, `incorrect`, `partial`, `citation`, `advice_boundary`, `notice_missing`,
`language`). The **root cause and fix** are written by hand after the generated table (blueprint §10.7:
retrieval miss / ranking / province or agency confusion / stale source / advice boundary / refusal error /
hallucination / citation / language / OCR / table extraction).

<!-- EVAL:START failure_analysis -->
_Not run yet._
<!-- EVAL:END failure_analysis -->

## Limitations

- The golden set is small; with this few questions a single item moves a rate by several points.
- The dev split (≈ 40%) is used to tune τ and top-k; headline numbers are on the test split only.
- The language detector is a transparent heuristic (script ratio, then Urdu vs English function words).
