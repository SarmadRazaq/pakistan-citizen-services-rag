"""Deterministic metrics (blueprint §10.2): retrieval, refusal, citations, notices, language, latency.

Everything here is pure: it takes API JSON (dicts) and golden items, and returns numbers. No LLM calls,
no network, so the whole retrieval grid scores for free and the same inputs always give the same output.

Definitions
- **Hit@k**: at least one relevant chunk (grade 2, see `relevance.py`) in the top k. Answerable items.
- **Recall@k**: share of the item's gold sources covered by a relevant chunk in the top k.
- **MRR@10**: 1 / rank of the first relevant chunk within the top 10, else 0.
- **nDCG@10**: linear gain (2 = right file and page/section, 1 = right file only), discount log2(rank+1).
  The ideal list holds max(#gold sources, #grade-2 chunks retrieved) grade-2 entries, then the grade-1
  chunks retrieved, so the score is at most 1 and a system that finds nothing scores 0.
- **Refusal**: a response is a refusal when its `outcome` is `refusal` (GLOSSARY.md: partial and degraded
  answers are answers). Cached runs from before `outcome` fall back to `answerable` and `refusal_reason`.
  Precision, recall and F1 treat "refuse" as the positive class, so recall is the share of unanswerable
  items refused and precision is hurt by false refusals. The false-refusal rate is reported over
  answerable items.
- **Citation validity**: share of *answered, answerable* items whose citations all point at gold-relevant
  chunks. An answer with no citation at all counts as invalid (an uncited claim cannot be checked).
- **Notices**: the expected notices (`not_advice`, `dated_information`, …) are all present.
- **Language match**: the detected answer language equals the expected one (`language.py`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from evals.golden import GoldenItem, GoldSource
from evals.language import detect_language
from evals.relevance import chunk_grade, is_relevant, matched_gold

HIT_KS = (1, 3, 5, 10)
RECALL_KS = (5, 10)
HEADLINE_NOTICES = ("not_advice", "dated_information", "referral")

Json = Mapping[str, Any]


# ---------------------------------------------------------------- ranking metrics on plain lists


def hit_at_k(relevant: Sequence[bool], k: int) -> float:
    return 1.0 if any(relevant[:k]) else 0.0


def recall_at_k(matches: Sequence[set[int]], n_gold: int, k: int) -> float:
    """`matches[i]` = indices of the gold sources the chunk at rank i+1 covers."""
    if n_gold <= 0:
        return 0.0
    covered: set[int] = set().union(*matches[:k]) if matches[:k] else set()
    return len(covered) / n_gold


def mrr_at_k(relevant: Sequence[bool], k: int = 10) -> float:
    for i, rel in enumerate(relevant[:k]):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def dcg(grades: Sequence[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(grades: Sequence[int], k: int = 10, n_gold: int = 1) -> float:
    """`grades` covers the whole retrieved list; only the top k count towards DCG."""
    n2 = max(n_gold, sum(1 for g in grades if g >= 2))
    n1 = sum(1 for g in grades if g == 1)
    ideal = ([2] * n2 + [1] * n1)[:k]
    idcg = dcg(ideal)
    return dcg(list(grades[:k])) / idcg if idcg > 0 else 0.0


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (the numpy default)."""
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(data[lo] + (data[hi] - data[lo]) * (pos - lo))


def refusal_prf(pairs: Iterable[tuple[bool, bool]]) -> dict[str, float | None]:
    """`pairs` = (should_refuse, refused). Positive class = refuse."""
    tp = fp = fn = tn = 0
    for should, did in pairs:
        if should and did:
            tp += 1
        elif not should and did:
            fp += 1
        elif should and not did:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is not None and recall is not None and precision + recall > 0:
        f1: float | None = 2 * precision * recall / (precision + recall)
    elif precision is not None and recall is not None:
        f1 = 0.0
    else:
        f1 = None
    total = tp + fp + fn + tn
    return {
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": f1,
        "refusal_accuracy": (tp + tn) / total if total else None,
        "false_refusal_rate": fp / (fp + tn) if fp + tn else None,
    }


# ---------------------------------------------------------------- per-item scoring from API JSON


def ranked_chunks(body: Json | None) -> list[Json]:
    """Chunks of a /search or /ask response, best first."""
    if not body:
        return []
    chunks = list((body.get("retrieval") or {}).get("chunks") or [])
    return sorted(chunks, key=lambda c: (c.get("rank") is None, c.get("rank") or 0))


def retrieval_scores(chunks: Sequence[Json], golds: Sequence[GoldSource]) -> dict[str, float]:
    grades = [chunk_grade(c, golds) for c in chunks]
    relevant = [g == 2 for g in grades]
    matches = [matched_gold(c, golds) for c in chunks]
    out: dict[str, float] = {f"hit@{k}": hit_at_k(relevant, k) for k in HIT_KS}
    out.update({f"recall@{k}": recall_at_k(matches, len(golds), k) for k in RECALL_KS})
    out["mrr@10"] = mrr_at_k(relevant, 10)
    out["ndcg@10"] = ndcg_at_k(grades, 10, n_gold=len(golds))
    return out


def is_refusal(ask: Json) -> bool:
    """A Refusal (GLOSSARY.md); partial and degraded answers are not refusals."""
    if "outcome" in ask:
        return bool(ask["outcome"] == "refusal")
    reason = ask.get("refusal_reason")  # cached runs from before `outcome` existed
    return ask.get("answerable") is False or (reason is not None and reason != "partial")


def _citation_chunk(citation: Json, by_id: Mapping[str, Json]) -> dict[str, Any]:
    """A citation as a chunk-like dict; the retrieved chunk with the same id fills in page spans."""
    merged: dict[str, Any] = dict(by_id.get(str(citation.get("chunk_id")), {}))
    for key in ("file_name", "page", "section"):
        if citation.get(key) is not None:
            merged[key] = citation[key]
    return merged


def citations_valid(ask: Json, golds: Sequence[GoldSource]) -> bool:
    citations = list(ask.get("citations") or [])
    if not citations:
        return False
    by_id = {str(c.get("chunk_id")): c for c in ranked_chunks(ask)}
    return all(is_relevant(_citation_chunk(c, by_id), golds) for c in citations)


def notice_codes(ask: Json) -> list[str]:
    return sorted({str(n.get("code")) for n in ask.get("notices") or [] if n.get("code")})


@dataclass
class ItemScore:
    """One golden item under one config. Every metric is None when it does not apply."""

    config: str
    id: str
    type: str
    language: str
    answer_language: str
    split: str
    answerable: bool
    expected_notices: list[str] = field(default_factory=list)
    retrieval: dict[str, float] | None = None
    context_hit: bool | None = None
    refused: bool | None = None
    refusal_reason: str | None = None
    citation_valid: bool | None = None
    n_citations: int = 0
    notices: list[str] = field(default_factory=list)
    notices_ok: bool | None = None
    notice_hits: dict[str, bool] = field(default_factory=dict)
    expected_language: str | None = None
    detected_language: str | None = None
    language_ok: bool | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    degraded: bool | None = None
    model: str | None = None
    judge_correctness: int | None = None
    judge_citation_support: str | None = None
    judge_refusal_appropriate: bool | None = None
    judge_advice_ok: bool | None = None
    judge_rationale: str | None = None
    error: str | None = None
    failure_category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_answer_language(item: GoldenItem, requested: str) -> str:
    return requested if requested in ("en", "ur", "roman_ur") else item.answer_language


def _base(config: str, item: GoldenItem) -> ItemScore:
    return ItemScore(
        config=config,
        id=item.id,
        type=item.type,
        language=item.language,
        answer_language=item.answer_language,
        split=item.split,
        answerable=item.answerable,
        expected_notices=list(item.expected_notices),
    )


def _language(score: ItemScore, answer: str, expected: str) -> None:
    if answer.strip():
        score.expected_language = expected
        score.detected_language = detect_language(answer)
        score.language_ok = score.detected_language == expected


def score_service_item(
    config: str,
    item: GoldenItem,
    search: Json | None,
    ask: Json | None,
    *,
    requested_language: str = "auto",
    client_latency_ms: float | None = None,
    error: str | None = None,
) -> ItemScore:
    score = _base(config, item)
    score.error = error
    if item.answerable:
        chunks = ranked_chunks(search) or ranked_chunks(ask)
        if search is not None or ask is not None:
            score.retrieval = retrieval_scores(chunks, item.gold_sources)
        used = [c for c in ranked_chunks(ask) if c.get("used_in_context")]
        if used:
            score.context_hit = any(is_relevant(c, item.gold_sources) for c in used)
    if ask is None:
        return score

    score.refused = is_refusal(ask)
    score.refusal_reason = ask.get("refusal_reason")
    score.n_citations = len(ask.get("citations") or [])
    if item.answerable and not score.refused:
        score.citation_valid = citations_valid(ask, item.gold_sources)
    score.notices = notice_codes(ask)
    if item.expected_notices:
        score.notice_hits = {code: code in score.notices for code in item.expected_notices}
        score.notices_ok = all(score.notice_hits.values())
    _language(score, str(ask.get("answer") or ""), expected_answer_language(item, requested_language))
    latency = (ask.get("latency_ms") or {}).get("total")
    score.latency_ms = float(latency) if isinstance(latency, int | float) else client_latency_ms
    usage = ask.get("usage") or {}
    score.input_tokens = usage.get("input_tokens")
    score.output_tokens = usage.get("output_tokens")
    score.degraded = ask.get("degraded")
    score.model = (ask.get("provider") or {}).get("model")
    return score


def score_closed_book_item(
    config: str,
    item: GoldenItem,
    answer: str | None,
    *,
    model: str | None,
    latency_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error: str | None = None,
) -> ItemScore:
    score = _base(config, item)
    score.error = error
    score.model = model
    if answer is None:
        return score
    _language(score, answer, item.answer_language)
    score.latency_ms = latency_ms
    score.input_tokens = input_tokens
    score.output_tokens = output_tokens
    return score


def failure_category(score: ItemScore, *, closed_book: bool = False) -> str:
    """A short label for per_question.csv and the failure analysis (blueprint §10.7)."""
    if score.error:
        return "error"
    if closed_book:
        if score.judge_correctness is None:
            return "unjudged"
        return {2: "ok", 1: "partial"}.get(score.judge_correctness, "incorrect")
    if score.refused is None:
        return "error"
    if not score.answerable:
        return "ok" if score.refused else "refusal_error"
    if score.refused:
        return "false_refusal"
    r = score.retrieval or {}
    if r and r.get("hit@10", 0.0) == 0.0:
        return "retrieval_miss"
    if score.context_hit is False or (score.context_hit is None and r and r.get("hit@5", 0.0) == 0.0):
        return "ranking"
    if score.judge_correctness == 0:
        return "incorrect"
    if score.citation_valid is False:
        return "citation"
    if score.judge_advice_ok is False or score.notices_ok is False:
        return "advice_boundary" if "not_advice" in score.expected_notices else "notice_missing"
    if score.language_ok is False:
        return "language"
    if score.judge_correctness == 1:
        return "partial"
    return "ok"


# ---------------------------------------------------------------- aggregation


def _mean(values: Iterable[float | int | bool | None]) -> float | None:
    data = [float(v) for v in values if v is not None]
    return sum(data) / len(data) if data else None


def _rate(values: Iterable[bool | None]) -> float | None:
    return _mean(values)


def aggregate(scores: Sequence[ItemScore]) -> dict[str, Any]:
    answerable = [s for s in scores if s.answerable]
    out: dict[str, Any] = {
        "n": len(scores),
        "n_answerable": len(answerable),
        "n_unanswerable": len(scores) - len(answerable),
        "errors": sum(1 for s in scores if s.error),
    }
    with_retrieval = [s.retrieval for s in answerable if s.retrieval is not None]
    keys = [f"hit@{k}" for k in HIT_KS] + [f"recall@{k}" for k in RECALL_KS] + ["mrr@10", "ndcg@10"]
    for key in keys:
        out[key] = _mean(r[key] for r in with_retrieval)
    out["context_hit"] = _rate(s.context_hit for s in answerable)
    out.update(refusal_prf((not s.answerable, s.refused) for s in scores if s.refused is not None))
    out["citation_validity"] = _rate(s.citation_valid for s in scores)
    out["notices_ok"] = _rate(s.notices_ok for s in scores)
    for code in HEADLINE_NOTICES:
        out[f"notice_{code}"] = _rate(s.notice_hits.get(code) for s in scores)
    out["language_match"] = _rate(s.language_ok for s in scores)
    latencies = [s.latency_ms for s in scores if s.latency_ms is not None]
    out["latency_p50_ms"] = percentile(latencies, 50)
    out["latency_p95_ms"] = percentile(latencies, 95)
    out["input_tokens_mean"] = _mean(s.input_tokens for s in scores)
    out["output_tokens_mean"] = _mean(s.output_tokens for s in scores)
    out["degraded_rate"] = _rate(s.degraded for s in scores)
    correctness = [s.judge_correctness for s in answerable if s.judge_correctness is not None]
    out["judge_correctness_mean"] = _mean(correctness)
    out["judge_correct_rate"] = _mean(c == 2 for c in correctness) if correctness else None
    support = [s.judge_citation_support for s in scores if s.judge_citation_support in ("yes", "no")]
    out["judge_citation_support"] = _mean(v == "yes" for v in support) if support else None
    out["judge_refusal_appropriate"] = _rate(s.judge_refusal_appropriate for s in scores)
    out["judge_advice_ok"] = _rate(s.judge_advice_ok for s in scores)
    out["judged"] = sum(1 for s in scores if s.judge_correctness is not None)
    return out


def _grouped(scores: Sequence[ItemScore], key: Callable[[ItemScore], str]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[ItemScore]] = {}
    for s in scores:
        groups.setdefault(key(s), []).append(s)
    return {name: aggregate(group) for name, group in sorted(groups.items())}


def breakdown(scores: Sequence[ItemScore]) -> dict[str, Any]:
    """Overall, per type, per language (en / ur / roman_ur separately) and per split."""
    return {
        "overall": aggregate(scores),
        "by_type": _grouped(scores, lambda s: s.type),
        "by_language": _grouped(scores, lambda s: s.language),
        "by_split": _grouped(scores, lambda s: s.split),
        "failures": _count(s.failure_category for s in scores),
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items()))
