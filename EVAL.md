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
- Run: `approved-test-2026-11-27` at 2026-11-27T11:32:11+00:00 against `http://127.0.0.1:8000`
- Judge: `gpt-4o` (prompt `judge-v1`), enabled
- Configs: `R4_hybrid_rerank`, `R1_dense_only`, `R5_hybrid_no_rerank`, `G2_gpt4o_mini_generator`, `CB_closed_book`
- Command: `python -m evals.run --split test --configs R4_hybrid_rerank R1_dense_only R5_hybrid_no_rerank G2_gpt4o_mini_generator CB_closed_book --concurrency 4 --run-name approved-test-2026-11-27`
<!-- EVAL:END run_info -->

## Golden set

`evals/golden_set.jsonl`, schema in `evals/golden.py` (blueprint §10.1). Gold labels are at
**document + page/section** level, so they stay valid across chunking configs. Only items with
`review_status: approved` are evaluated unless `--include-drafts` is passed; `dropped` items never run.

<!-- EVAL:START golden_set -->
File `evals/golden_set.jsonl`, sha256 `9fe657b81af5755b…`: **48 items** in the file, **33 evaluated** in this run (split `test`, drafts excluded).

| Type | Items |
|---|---|
| advice_boundary | 2 |
| ambiguity | 1 |
| contradiction | 4 |
| multi_part | 1 |
| paraphrase | 1 |
| single_fact | 19 |
| unanswerable_absent | 2 |
| unanswerable_near_miss | 3 |

| Language | Items |
|---|---|
| en | 20 |
| roman_ur | 8 |
| ur | 5 |

| Split | Items |
|---|---|
| test | 33 |

| Review status | Items |
|---|---|
| approved | 48 |
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
| Config | n | Hit@1 | Hit@5 | Hit@10 | R@10 | MRR@10 | nDCG@10 | Refusal F1 | False refusal | Citation valid | Notices | Lang match | Judge corr. (0-2) | Judge = 2 | p50 ms | p95 ms | Overrides |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R4_hybrid_rerank** (baseline) | 33 | 0.571 | 0.893 | 0.929 | 0.869 | 0.701 | 0.652 | 0.667 | 0.179 | 0.609 | 0.000 | 1.000 | 0.96 | 0.429 | 11300 | 18258 | none |
| R1_dense_only | 33 | 0.607 | 0.929 | 0.929 | 0.851 | 0.726 | 0.665 | 0.909 | 0.036 | 0.667 | 0.000 | 1.000 | 0.89 | 0.393 | 5823 | 17174 | honoured |
| R5_hybrid_no_rerank | 33 | 0.536 | 0.857 | 0.929 | 0.839 | 0.684 | 0.635 | 0.727 | 0.071 | 0.615 | 0.000 | 0.970 | 1.00 | 0.429 | 5461 | 16320 | honoured |
| G2_gpt4o_mini_generator | 33 | 0.571 | 0.893 | 0.929 | 0.869 | 0.701 | 0.652 | 0.526 | 0.321 | 0.684 | 0.000 | 1.000 | 0.96 | 0.393 | 12916 | 18655 | honoured |
| CB_closed_book | 28 | – | – | – | – | – | – | – | – | – | – | 1.000 | 0.29 | 0.107 | – | – | closed-book (no sources) |
<!-- EVAL:END config_comparison -->

### Winner and what it means (run `approved-test-2026-11-27`, 33 approved test items)

**By the rule fixed in advance, `R1_dense_only` wins:** MRR@10 0.726 against 0.701 for the `R4_hybrid_rerank`
baseline. It also refuses far less often when it should answer (false refusal 0.036 against 0.179) and has the
better refusal F1 (0.909 against 0.667).

The rule only looks at retrieval, and the other columns pull the other way:
- **More wrong answers:** dense-only gives 12 answers the judge marks incorrect, against 6 for the baseline.
- **Lower judge correctness:** 0.89 against 0.96.
- **Small margin:** the MRR gap is about one question on 33, well inside the noise of a set this size.

What the numbers support:
- **The reranker earns its cost in answer quality, not in ranking.** Without it, the relevance gate runs on dense
  cosine, which lets weaker contexts through: more answers, and more wrong ones.
- **The service default stays `R4_hybrid_rerank`** until a larger golden set separates the two. This is a stated
  deviation from the winner rule, made because the rule turned out to measure the wrong thing for this decision.
  The next revision of the rule should rank by judge correctness subject to a false-refusal ceiling.

### Threshold tuning and the two fixes between runs

The first full run (`baseline-2026-11-27`, 48 draft items) refused **41.5%** of answerable questions with the
baseline config. Two causes were found and fixed before the final run:

1. **Quote check vs Markdown tables** ([app/generation/answer.py](app/generation/answer.py)). Table chunks are
   wide Markdown tables. A verbatim quote of a cell scored below 90 against text full of `|` characters, so
   correct citations were dropped and the answer refused. The comparison now ignores table markup on both sides.
   The threshold and the number check are unchanged.
2. **Reranker against the original question only** ([app/retrieval/retriever.py](app/retrieval/retriever.py)).
   Roman Urdu questions scored 0.006 to 0.23 against the correct English passages. The score is now the maximum
   over the query variants (original, English and Urdu-script rewrites). On the dev split, two Roman Urdu
   questions rose from 0.006 to 0.92 and from 0.065 to 0.995.

`tau_refuse` was then re-tuned **on the dev split only** (15 items, two of them unanswerable):
- The highest-scoring unanswerable dev question scored 0.162.
- The lowest answerable one within reach scored 0.205.
- Any threshold between them is equally good on dev; **0.2** was chosen (was 0.3).
- The test split was not looked at for this choice.

| Run (baseline config, all 48 items) | False refusal | Refusal F1 | Judge corr. (0-2) | Citation valid |
|---|---|---|---|---|
| Before the fixes (`baseline-2026-11-27`) | 0.415 | 0.452 | 0.83 | 0.583 |
| After the fixes and `tau_refuse` 0.2 (`final-2026-11-27`) | 0.171 | 0.667 | 0.98 | 0.471 |

Citation validity fell because more questions are now answered, and the new answers often cite a second,
legitimate source that is not in the gold labels (see the failure analysis).


### By question type

<!-- EVAL:START by_type -->
| Config | Type | n | Hit@5 | MRR@10 | Refusal F1 | Citation valid | Lang match | Judge corr. (0-2) |
|---|---|---|---|---|---|---|---|---|
| R4_hybrid_rerank | advice_boundary | 2 | 0.500 | 0.167 | – | – | 1.000 | 0.00 |
| R4_hybrid_rerank | ambiguity | 1 | 0.000 | 0.000 | – | 0.000 | 1.000 | 0.00 |
| R4_hybrid_rerank | contradiction | 4 | 1.000 | 0.833 | – | 0.750 | 1.000 | 1.00 |
| R4_hybrid_rerank | multi_part | 1 | 1.000 | 0.500 | – | 0.000 | 1.000 | 0.00 |
| R4_hybrid_rerank | paraphrase | 1 | 1.000 | 0.500 | – | – | 1.000 | 0.00 |
| R4_hybrid_rerank | single_fact | 19 | 0.947 | 0.787 | – | 0.647 | 1.000 | 1.21 |
| R4_hybrid_rerank | unanswerable_absent | 2 | – | – | 1.000 | – | 1.000 | – |
| R4_hybrid_rerank | unanswerable_near_miss | 3 | – | – | 1.000 | – | 1.000 | – |
| R1_dense_only | advice_boundary | 2 | 0.500 | 0.500 | – | 0.500 | 1.000 | 0.00 |
| R1_dense_only | ambiguity | 1 | 0.000 | 0.000 | – | – | 1.000 | 0.00 |
| R1_dense_only | contradiction | 4 | 1.000 | 0.688 | – | 0.750 | 1.000 | 0.25 |
| R1_dense_only | multi_part | 1 | 1.000 | 0.500 | – | 0.000 | 1.000 | 0.00 |
| R1_dense_only | paraphrase | 1 | 1.000 | 0.333 | – | 0.000 | 1.000 | 0.00 |
| R1_dense_only | single_fact | 19 | 1.000 | 0.829 | – | 0.737 | 1.000 | 1.26 |
| R1_dense_only | unanswerable_absent | 2 | – | – | 1.000 | – | 1.000 | – |
| R1_dense_only | unanswerable_near_miss | 3 | – | – | 1.000 | – | 1.000 | – |
| R5_hybrid_no_rerank | advice_boundary | 2 | 0.500 | 0.500 | – | 0.500 | 1.000 | 0.00 |
| R5_hybrid_no_rerank | ambiguity | 1 | 0.000 | 0.000 | – | 0.000 | 1.000 | 0.00 |
| R5_hybrid_no_rerank | contradiction | 4 | 1.000 | 0.800 | – | 0.667 | 1.000 | 0.75 |
| R5_hybrid_no_rerank | multi_part | 1 | 1.000 | 0.500 | – | 0.000 | 1.000 | 0.00 |
| R5_hybrid_no_rerank | paraphrase | 1 | 1.000 | 0.200 | – | 0.000 | 1.000 | 2.00 |
| R5_hybrid_no_rerank | single_fact | 19 | 0.895 | 0.750 | – | 0.722 | 1.000 | 1.21 |
| R5_hybrid_no_rerank | unanswerable_absent | 2 | – | – | 0.667 | – | 0.500 | – |
| R5_hybrid_no_rerank | unanswerable_near_miss | 3 | – | – | 1.000 | – | 1.000 | – |
| G2_gpt4o_mini_generator | advice_boundary | 2 | 0.500 | 0.167 | – | – | 1.000 | 0.00 |
| G2_gpt4o_mini_generator | ambiguity | 1 | 0.000 | 0.000 | – | – | 1.000 | 0.00 |
| G2_gpt4o_mini_generator | contradiction | 4 | 1.000 | 0.833 | – | 1.000 | 1.000 | 0.00 |
| G2_gpt4o_mini_generator | multi_part | 1 | 1.000 | 0.500 | – | 0.000 | 1.000 | 1.00 |
| G2_gpt4o_mini_generator | paraphrase | 1 | 1.000 | 0.500 | – | – | 1.000 | 0.00 |
| G2_gpt4o_mini_generator | single_fact | 19 | 0.947 | 0.787 | – | 0.706 | 1.000 | 1.37 |
| G2_gpt4o_mini_generator | unanswerable_absent | 2 | – | – | 1.000 | – | 1.000 | – |
| G2_gpt4o_mini_generator | unanswerable_near_miss | 3 | – | – | 1.000 | – | 1.000 | – |
| CB_closed_book | advice_boundary | 2 | – | – | – | – | 1.000 | 0.00 |
| CB_closed_book | ambiguity | 1 | – | – | – | – | 1.000 | 0.00 |
| CB_closed_book | contradiction | 4 | – | – | – | – | 1.000 | 0.00 |
| CB_closed_book | multi_part | 1 | – | – | – | – | 1.000 | 0.00 |
| CB_closed_book | paraphrase | 1 | – | – | – | – | 1.000 | 2.00 |
| CB_closed_book | single_fact | 19 | – | – | – | – | 1.000 | 0.32 |
<!-- EVAL:END by_type -->

## Generator comparison

`G2_gpt4o_mini_generator` swaps the generator for `gpt-4o-mini`, with the same retrieval as the baseline. Both use
the `gpt-4o` judge.

| Generator | Judge corr. (0-2) | False refusal | Refusal F1 | Citation valid | Lang match | p95 ms |
|---|---|---|---|---|---|---|
| `gpt-4.1-mini` (baseline) | 0.96 | 0.179 | 0.667 | 0.609 | 1.000 | 18,258 |
| `gpt-4o-mini` | 0.96 | 0.321 | 0.526 | 0.684 | 1.000 | 18,655 |

Selection rule (fixed in advance): maximise judge correctness, subject to faithfulness ≥ 0.85, refusal accuracy
≥ 0.8 and answer-language accuracy ≥ 0.9.
- **Neither model meets the refusal constraint.**
- **Faithfulness is not measured,** because RAGAS is not installed. Citation validity is the nearest measured
  proxy.
- **On what is measured, `gpt-4.1-mini` stays:** the same judge correctness, with roughly half the false refusals.

## Judge calibration

The judge is `gpt-4o`: a different, larger model than the generators (`gpt-4.1-mini`, `gpt-4o-mini`). It is the same vendor, so some self-preference bias remains possible, which is what the calibration below measures. To check it
(blueprint §10.5):

1. `python -m evals.calibration export --results evals/results/<timestamp>` writes a stratified sample
   of 30 baseline answers to `evals/human_labels.csv`, **without** the judge's scores.
2. Label every row: `human_correctness` 2 / 1 / 0, `human_citation_supported` y / n.
3. `python -m evals.calibration score --results evals/results/<timestamp>` reports % agreement and
   Cohen's κ (target ≥ 0.6) and refreshes the table below.

<!-- EVAL:START judge_calibration -->
_No human labels yet._ 30 answers from `approved-test-2026-11-27` are exported to
`evals/human_labels.csv`, waiting for the student's labels. Until they are scored, the judge's reliability is
**unverified**, and every judge column above should be read with that caveat.
<!-- EVAL:END judge_calibration -->

## Multilingual

Urdu and Roman Urdu are reported separately from English, never averaged into one number.

<!-- EVAL:START multilingual -->
| Config | Language | n | Hit@5 | MRR@10 | Refusal F1 | Citation valid | Lang match | Judge corr. (0-2) |
|---|---|---|---|---|---|---|---|---|
| R4_hybrid_rerank | en | 20 | 0.875 | 0.658 | 0.800 | 0.500 | 1.000 | 1.12 |
| R4_hybrid_rerank | ur | 5 | 1.000 | 0.900 | – | 1.000 | 1.000 | 1.40 |
| R4_hybrid_rerank | roman_ur | 8 | 0.857 | 0.655 | 0.500 | 0.600 | 1.000 | 0.29 |
| R1_dense_only | en | 20 | 0.938 | 0.693 | 0.889 | 0.600 | 1.000 | 1.00 |
| R1_dense_only | ur | 5 | 1.000 | 0.850 | – | 1.000 | 1.000 | 1.20 |
| R1_dense_only | roman_ur | 8 | 0.857 | 0.714 | 1.000 | 0.571 | 1.000 | 0.43 |
| R5_hybrid_no_rerank | en | 20 | 0.875 | 0.627 | 0.889 | 0.467 | 1.000 | 1.12 |
| R5_hybrid_no_rerank | ur | 5 | 0.800 | 0.725 | – | 1.000 | 1.000 | 1.00 |
| R5_hybrid_no_rerank | roman_ur | 8 | 0.857 | 0.786 | 0.000 | 0.667 | 0.875 | 0.71 |
| G2_gpt4o_mini_generator | en | 20 | 0.875 | 0.658 | 0.571 | 0.600 | 1.000 | 0.88 |
| G2_gpt4o_mini_generator | ur | 5 | 1.000 | 0.900 | – | 1.000 | 1.000 | 1.60 |
| G2_gpt4o_mini_generator | roman_ur | 8 | 0.857 | 0.655 | 0.500 | 0.600 | 1.000 | 0.71 |
| CB_closed_book | en | 16 | – | – | – | – | 1.000 | 0.31 |
| CB_closed_book | ur | 5 | – | – | – | – | 1.000 | 0.20 |
| CB_closed_book | roman_ur | 7 | – | – | – | – | 1.000 | 0.29 |

**Cross-lingual gap** (language group minus the English group; the blueprint's paired version needs translation links in the golden set):

| Config | Language | Δ Hit@5 vs en | Δ Judge corr. vs en | Δ Lang match vs en |
|---|---|---|---|---|
| R4_hybrid_rerank | ur | +0.125 | +0.275 | +0.000 |
| R4_hybrid_rerank | roman_ur | -0.018 | -0.839 | +0.000 |
| R1_dense_only | ur | +0.062 | +0.200 | +0.000 |
| R1_dense_only | roman_ur | -0.080 | -0.571 | +0.000 |
| R5_hybrid_no_rerank | ur | -0.075 | -0.125 | +0.000 |
| R5_hybrid_no_rerank | roman_ur | -0.018 | -0.411 | -0.125 |
| G2_gpt4o_mini_generator | ur | +0.125 | +0.725 | +0.000 |
| G2_gpt4o_mini_generator | roman_ur | -0.018 | -0.161 | +0.000 |
| CB_closed_book | ur | – | -0.112 | +0.000 |
| CB_closed_book | roman_ur | – | -0.027 | +0.000 |
<!-- EVAL:END multilingual -->

## Contamination

The closed-book baseline asks the generator the same questions **without any sources**. A high
closed-book score means the model already knows the answer from pre-training; the RAG gain over it is
what retrieval contributes.

<!-- EVAL:START contamination -->
| Config | Model | Answerable n | Judge corr. (0-2) | Judge = 2 | RAG baseline − closed-book |
|---|---|---|---|---|---|
| CB_closed_book | gpt-4.1-mini | 28 | 0.29 | 0.107 | +0.68 |
| R4_hybrid_rerank | gpt-4.1-mini | 28 | 0.96 | 0.429 |  |
| R1_dense_only | gpt-4.1-mini | 28 | 0.89 | 0.393 |  |
| R5_hybrid_no_rerank | gpt-4.1-mini | 28 | 1.00 | 0.429 |  |
| G2_gpt4o_mini_generator | gpt-4o-mini | 28 | 0.96 | 0.393 |  |
<!-- EVAL:END contamination -->

## Failure analysis

Each failed item is categorized automatically (`error`, `false_refusal`, `refusal_error`,
`retrieval_miss`, `ranking`, `incorrect`, `partial`, `citation`, `advice_boundary`, `notice_missing`,
`language`). The **root cause and fix** are written by hand after the generated table (blueprint §10.7:
retrieval miss / ranking / province or agency confusion / stale source / advice boundary / refusal error /
hallucination / citation / language / OCR / table extraction).

<!-- EVAL:START failure_analysis -->
| Config | Failures by category | ok |
|---|---|---|
| R4_hybrid_rerank | citation: 3, false_refusal: 5, incorrect: 6, partial: 3, ranking: 1, retrieval_miss: 1 | 14 |
| R1_dense_only | citation: 3, false_refusal: 1, incorrect: 12, partial: 2, retrieval_miss: 1 | 14 |
| R5_hybrid_no_rerank | citation: 3, false_refusal: 2, incorrect: 7, partial: 3, ranking: 1, refusal_error: 1, retrieval_miss: 2 | 14 |
| G2_gpt4o_mini_generator | citation: 4, false_refusal: 9, incorrect: 2, partial: 3, ranking: 1 | 14 |
| CB_closed_book | incorrect: 23, partial: 2 | 3 |

| Config | Item | Type | Lang | Category | Detail (judge rationale / error) | Root cause & fix |
|---|---|---|---|---|---|---|
| R4_hybrid_rerank | q_nadra_002 | single_fact | roman_ur | incorrect | The system answer incorrectly states that a marriage certificate is mandatory without mentioning the exception if the spouse has already added the applicant to  |  |
| R4_hybrid_rerank | q_nadra_003 | single_fact | ur | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| R4_hybrid_rerank | q_nadra_005 | advice_boundary | roman_ur | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| R4_hybrid_rerank | q_nadra_008 | single_fact | en | ranking | The system answer omits key requirements such as the S1 form or attested documents from the Embassy/Mission and includes unsupported details. |  |
| R4_hybrid_rerank | q_pp_002 | single_fact | roman_ur | incorrect | The system answer provides conflicting information about the delivery time for urgent passports, which contradicts the reference answer. |  |
| R4_hybrid_rerank | q_pp_005 | single_fact | en | citation | The system answer correctly states the extra fees for first and second-time passport loss, matching the reference answer. |  |
| R4_hybrid_rerank | q_nadra_011 | single_fact | roman_ur | partial | The system answer correctly states the fees but omits the processing times for each priority level. |  |
| R4_hybrid_rerank | q_tx_001 | single_fact | en | citation | The system answer correctly states the income tax return deadline for individuals as on or before 30th September, matching the reference answer. |  |
| R4_hybrid_rerank | q_tx_002 | single_fact | roman_ur | incorrect | The system answer incorrectly states the surcharge amount as Rs 25,000 instead of Rs 1,000 for individuals. |  |
| R4_hybrid_rerank | q_tx_005 | multi_part | en | incorrect | The system answer incorrectly states the surcharge for a company as one hundred thousand rupees instead of the correct amount of twenty thousand rupees. |  |
| R4_hybrid_rerank | q_tx_006 | advice_boundary | roman_ur | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| R4_hybrid_rerank | q_rc_004 | contradiction | en | incorrect | The system answer contradicts the reference by not mentioning that a representative can file a complaint, as stated on the 'Making a complaint' page. |  |
| R4_hybrid_rerank | q_rc_006 | single_fact | en | citation | The system answer includes all key facts from the reference answer without contradiction. |  |
| R4_hybrid_rerank | q_rc_007 | paraphrase | en | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| R4_hybrid_rerank | q_cp_001 | ambiguity | en | retrieval_miss | The system answer incorrectly addresses the role of the Wafaqi Mohtasib instead of the provincial consumer courts as specified in the reference answer. |  |
| R4_hybrid_rerank | q_cp_002 | single_fact | roman_ur | partial | The system answer correctly states the 15-day notice requirement but omits the 30-day filing period after the cause of action. |  |
| R4_hybrid_rerank | q_cp_004 | single_fact | ur | partial | The system answer includes the key fact about the 15-day notice but adds extra details not present in the reference. |  |
| R4_hybrid_rerank | q_cp_005 | contradiction | en | incorrect | The system answer incorrectly states that appeals go to the Sindh High Court, contradicting the reference answer which specifies the District and Sessions Court |  |
| R4_hybrid_rerank | q_dl_001 | single_fact | en | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| R1_dense_only | q_nadra_002 | single_fact | roman_ur | incorrect | The system answer omits the key detail that no document is needed if the spouse has already added the applicant to their family tree. |  |
| R1_dense_only | q_nadra_003 | single_fact | ur | incorrect | The system answer includes incorrect details about the required documents and procedures for changing the date of birth by more than five years. |  |
| R1_dense_only | q_nadra_005 | advice_boundary | roman_ur | incorrect | The system answer includes unnecessary details and fails to provide the required disclaimer and referral. |  |
| R1_dense_only | q_nadra_008 | single_fact | en | citation | The system answer includes all key facts from the reference answer and does not contradict it. |  |
| R1_dense_only | q_pp_002 | single_fact | roman_ur | partial | The system answer correctly states the issuance time but omits the detail about courier time to remote areas and missions. |  |
| R1_dense_only | q_pp_003 | single_fact | ur | incorrect | The system answer incorrectly states the delivery time as ten working days, contradicting the reference answer of twenty-one working days. |  |
| R1_dense_only | q_pp_005 | single_fact | en | citation | The system answer correctly states the fees for first and second-time passport loss but omits the procedure for third-time loss. |  |
| R1_dense_only | q_pp_007 | contradiction | en | partial | The system answer correctly states that any citizen can apply at any passport office, but it omits the detail about the first passport being issued at the CNIC  |  |
| R1_dense_only | q_nadra_012 | contradiction | en | incorrect | The system answer contradicts the reference by providing incorrect processing time and cost for a Smart NIC renewal. |  |
| R1_dense_only | q_tx_001 | single_fact | en | citation | The system answer correctly states the income tax return deadline for individuals as on or before 30th September, matching the reference answer. |  |
| R1_dense_only | q_tx_002 | single_fact | roman_ur | incorrect | The system answer incorrectly states the surcharge amount as Rs 25,000 for individuals, contradicting the reference answer which states it is Rs 1,000. |  |
| R1_dense_only | q_tx_005 | multi_part | en | incorrect | The system answer incorrectly states the surcharge for a company as one hundred thousand Rupees instead of twenty thousand Rupees. |  |
| R1_dense_only | q_tx_006 | advice_boundary | roman_ur | retrieval_miss | The system answer incorrectly advises the user to appeal, which contradicts the reference guidance not to provide personal advice. |  |
| R1_dense_only | q_rc_001 | single_fact | en | incorrect | The system answer addresses the resolution time for complaints, not the time limit for filing them, which is the main point of the question. |  |
| R1_dense_only | q_rc_004 | contradiction | en | incorrect | The system answer contradicts the reference by not mentioning that only the aggrieved person or legal heirs can file a complaint, according to the FAQ. |  |
| R1_dense_only | q_rc_007 | paraphrase | en | incorrect | The system answer includes additional procedural details not present in the reference, which could mislead the user. |  |
| R1_dense_only | q_cp_001 | ambiguity | en | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| R1_dense_only | q_cp_002 | single_fact | roman_ur | incorrect | The system answer omits the requirement to file the claim within 30 days of the cause of action. |  |
| R1_dense_only | q_cp_005 | contradiction | en | incorrect | The system answer incorrectly states that appeals go to the Sindh High Court, contradicting the reference answer which specifies the District and Sessions Court |  |
| R5_hybrid_no_rerank | q_nadra_001 | single_fact | en | incorrect | The system answer incorrectly adds the requirement of a notarized affidavit, which is not mentioned in the reference answer. |  |
| R5_hybrid_no_rerank | q_nadra_002 | single_fact | roman_ur | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| R5_hybrid_no_rerank | q_nadra_003 | single_fact | ur | incorrect | The system answer incorrectly states that a verified computerized birth certificate is mandatory for changes over five years, which is not mentioned in the refe |  |
| R5_hybrid_no_rerank | q_nadra_005 | advice_boundary | roman_ur | incorrect | The system answer includes unnecessary details and fails to provide the required disclaimer and referral. |  |
| R5_hybrid_no_rerank | q_nadra_008 | single_fact | en | incorrect | The system answer incorrectly adds the requirement of a notarized undertaking for those over 18, which is not mentioned in the reference answer. |  |
| R5_hybrid_no_rerank | q_pp_003 | single_fact | ur | incorrect | The system answer contradicts the reference answer by stating the delivery time as ten working days instead of twenty-one. |  |
| R5_hybrid_no_rerank | q_pp_005 | single_fact | en | citation | The system answer correctly states the fees for first and second-time passport loss and provides additional information about subsequent losses, which does not  |  |
| R5_hybrid_no_rerank | q_nadra_012 | contradiction | en | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| R5_hybrid_no_rerank | q_tx_001 | single_fact | en | citation | The system answer correctly states the deadline for individuals but omits the deadline for companies. |  |
| R5_hybrid_no_rerank | q_tx_002 | single_fact | roman_ur | incorrect | The system answer incorrectly states the surcharge amount as Rs. 25,000 for individuals, contradicting the reference answer which states Rs. 1,000. |  |
| R5_hybrid_no_rerank | q_tx_004 | single_fact | ur | partial | The system answer correctly states the publication and validity period of the ATL but omits the detail about weekly updates. |  |
| R5_hybrid_no_rerank | q_tx_005 | multi_part | en | incorrect | The system answer incorrectly states the surcharge for a company as PKR 100,000 instead of PKR 20,000. |  |
| R5_hybrid_no_rerank | q_tx_006 | advice_boundary | roman_ur | retrieval_miss | The system answer incorrectly advises the user on appealing, which contradicts the reference guidance to avoid giving personal advice. |  |
| R5_hybrid_no_rerank | q_rc_004 | contradiction | en | partial | The system answer includes both perspectives but incorrectly states that a representative cannot file a complaint unless they are legal heirs, which is not alig |  |
| R5_hybrid_no_rerank | q_rc_007 | paraphrase | en | citation | The system answer correctly states that the Ombudsman deals with delays by federal agencies, aligning with the reference answer. |  |
| R5_hybrid_no_rerank | q_cp_001 | ambiguity | en | retrieval_miss | The system answer incorrectly addresses consumer complaints through the Wafaqi Mohtasib, which is not relevant to the provincial consumer courts mentioned in th |  |
| R5_hybrid_no_rerank | q_cp_002 | single_fact | roman_ur | partial | The system answer correctly states the requirement of a 15-day notice but omits the detail about filing the claim within 30 days of the cause of action. |  |
| R5_hybrid_no_rerank | q_cp_005 | contradiction | en | ranking | The system answer incorrectly states that appeals go to the Sindh High Court, contradicting the reference answer which specifies the District and Sessions Court |  |
| R5_hybrid_no_rerank | q_un_002 | unanswerable_absent | roman_ur | refusal_error | The system provided a list of hospitals which is not supported by the corpus, contradicting the reference answer that the information is not available. |  |
| G2_gpt4o_mini_generator | q_nadra_002 | single_fact | roman_ur | citation | The system answer includes the need for a marriage certificate and mentions the mobile app application, but omits details about the spouse's ID and the conditio |  |
| G2_gpt4o_mini_generator | q_nadra_003 | single_fact | ur | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |
| G2_gpt4o_mini_generator | q_nadra_005 | advice_boundary | roman_ur | false_refusal | The system declined to answer despite the question being answerable from the corpus. |  |

_41 more in per_question.csv._
<!-- EVAL:END failure_analysis -->

### Root causes, by hand (baseline config, approved test split)

| Category | Items | Root cause | Fix |
|---|---|---|---|
| Wrong number from the wrong source | q_tx_002 (said Rs 25,000, correct Rs 1,000), q_tx_005 | The right page (TX-02b) was retrieved at rank 1, but the answer took a surcharge figure from the Income Tax Ordinance (TX-04), which was also in context. Each quote was real, so the quote check passed. It does not check that a number written in a sentence appears in that sentence's own cited quote. | **Not yet fixed.** Check every number in an answer sentence against the quotes that sentence cites, and regenerate on a mismatch. |
| Wrong agency for an ambiguous question | q_cp_001 | Consumer-law chunks were retrieved, but the reranker put the ombudsman's "complaints" pages first, so the answer described the wrong forum. | **Not yet fixed.** Topic routing: when query understanding detects the `consumer_protection` Topic, filter to it, as is already done for provinces. |
| Contradiction shown one-sided | q_rc_004 | Both pages were in context (RC-03 at rank 2), but the answer used only the FAQ. The prompt's "present both sources" rule was not followed. | Prompt work, or a check that flags a contradiction-type question answered from a single Document. |
| False refusals of personal questions | q_nadra_005, q_tx_006 (advice-boundary) | A personal-case question ("mera case court mein hai…") retrieves poorly: the top chunks were the Passport Rules and the Income Tax Ordinance, so the gate refused. | Rewrite advice-request questions into the general rule they ask about before retrieval. |
| Citation of a legitimate extra source | several | The metric counts an answer valid only if every Citation points at a gold Source. Answers often add a second, correct source (the Ordinance, the Regulations). Gold labels were widened only where the second source was checked to state the same fact. | Labelling, not a model fault. |
| Notices never complete | advice-boundary items | `referral` can only be shown from a verified entry in `referrals.yaml`, and none is verified yet. | The student verifies the referral contacts. |

**Roman Urdu is the weakest group:** judge correctness 0.29, against 1.40 for Urdu script and 1.12 for English,
although retrieval is close (Hit@5 0.857). Answers in Roman Urdu are where generation fails most. That is the
first place to look in the next iteration.


## Limitations

- The golden set is small (48 items, 33 on test); a single item moves a rate by about 3 points.
- The items were drafted from the corpus text and reviewed against their sources with AI assistance
  (`reviewed_by: claude-assisted-review-2026-11-27`). The student's own review should replace that marker.
- Faithfulness (RAGAS) is not measured; `--ragas` needs `requirements-eval`.
- The dev split (≈ 40%) is used to tune τ and top-k; headline numbers are on the test split only.
- The language detector is a transparent heuristic (script ratio, then Urdu vs English function words).
