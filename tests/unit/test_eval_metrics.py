"""Deterministic eval metrics on hand-made rankings (evals/metrics.py, blueprint §10.2)."""

from __future__ import annotations

import math
from typing import Any

import pytest
from evals.golden import GoldenItem, GoldSource
from evals.language import detect_language
from evals.metrics import (
    ItemScore,
    aggregate,
    breakdown,
    citations_valid,
    failure_category,
    hit_at_k,
    is_refusal,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    refusal_prf,
    retrieval_scores,
    score_service_item,
)

FILE = "ID-05__nadra-registration-policy-6-0-2-en.pdf"


def _item(**over: Any) -> GoldenItem:
    data: dict[str, Any] = {
        "id": "q_t_001",
        "question": "What documents do I need?",
        "language": "en",
        "answer_language": "en",
        "type": "single_fact",
        "answerable": True,
        "gold_sources": [{"file_name": FILE, "pages": [13]}],
        "agencies": ["NADRA"],
        "jurisdictions": ["federal"],
        "expected_notices": [],
        "source_ids": ["ID-05"],
        "reference_answer": "A verified birth certificate.",
        "split": "test",
        "review_status": "approved",
        "author": "student",
        "reviewed_by": "student",
    }
    data.update(over)
    return GoldenItem.model_validate(data)


def _chunk(rank: int, page: int | None, file_name: str = FILE, **over: Any) -> dict[str, Any]:
    return {"rank": rank, "chunk_id": f"c{rank}", "file_name": file_name, "page": page, "text": "t", **over}


# ---------------------------------------------------------------- ranking metrics


def test_hit_at_k() -> None:
    rel = [False, True, False, True]
    assert [hit_at_k(rel, k) for k in (1, 2, 3, 10)] == [0.0, 1.0, 1.0, 1.0]
    assert hit_at_k([], 5) == 0.0


def test_mrr() -> None:
    assert mrr_at_k([False, True, False, True]) == 0.5
    assert mrr_at_k([False, False, True]) == pytest.approx(1 / 3)
    assert mrr_at_k([True]) == 1.0
    assert mrr_at_k([False] * 10 + [True]) == 0.0  # beyond rank 10


def test_recall_at_k() -> None:
    matches: list[set[int]] = [set(), {0}, set(), {1}, {0}]
    assert recall_at_k(matches, 2, 1) == 0.0
    assert recall_at_k(matches, 2, 2) == 0.5
    assert recall_at_k(matches, 2, 4) == 1.0
    assert recall_at_k(matches, 3, 10) == pytest.approx(2 / 3)
    assert recall_at_k([], 2, 5) == 0.0


def test_ndcg_exact_values() -> None:
    # grades 0, 2, 1 with one gold source: DCG = 2/log2(3) + 1/log2(4); ideal [2, 1] = 2/1 + 1/log2(3)
    expected = (2 / math.log2(3) + 1 / math.log2(4)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k([0, 2, 1], 10, n_gold=1) == pytest.approx(expected)
    assert ndcg_at_k([2, 1, 0], 10, n_gold=1) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], 10, n_gold=1) == 0.0
    # two gold sources but only one found at rank 1: ideal assumes two grade-2 chunks
    assert ndcg_at_k([2, 0], 10, n_gold=2) == pytest.approx(2 / (2 + 2 / math.log2(3)))
    # a relevant chunk beyond the cut-off does not count, but still shapes the ideal list
    assert ndcg_at_k([0, 2], 1, n_gold=1) == 0.0


def test_retrieval_scores_from_chunks() -> None:
    golds = [GoldSource(file_name=FILE, pages=[13])]
    chunks = [_chunk(1, 2), _chunk(2, 13), _chunk(3, 13, file_name="other.pdf")]
    s = retrieval_scores(chunks, golds)
    assert s["hit@1"] == 0.0 and s["hit@3"] == 1.0 and s["hit@10"] == 1.0
    assert s["mrr@10"] == 0.5
    assert s["recall@5"] == 1.0
    expected = (1 / 1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))  # grades 1, 2, 0
    assert s["ndcg@10"] == pytest.approx(expected)


def test_percentile_linear() -> None:
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


# ---------------------------------------------------------------- refusal


def test_refusal_prf() -> None:
    # (should refuse, refused): 2 TP, 1 FN, 1 FP, 2 TN
    pairs = [(True, True), (True, True), (True, False), (False, True), (False, False), (False, False)]
    r = refusal_prf(pairs)
    assert r["refusal_precision"] == pytest.approx(2 / 3)
    assert r["refusal_recall"] == pytest.approx(2 / 3)
    assert r["refusal_f1"] == pytest.approx(2 / 3)
    assert r["false_refusal_rate"] == pytest.approx(1 / 3)
    assert r["refusal_accuracy"] == pytest.approx(4 / 6)


def test_refusal_prf_edge_cases() -> None:
    assert refusal_prf([(False, False)])["refusal_f1"] is None  # no refusals expected or made
    r = refusal_prf([(True, False), (False, True)])
    assert r["refusal_precision"] == 0.0 and r["refusal_recall"] == 0.0 and r["refusal_f1"] == 0.0


def test_is_refusal() -> None:
    assert is_refusal({"answerable": False, "refusal_reason": None})
    assert is_refusal({"answerable": True, "refusal_reason": "insufficient_context"})
    assert not is_refusal({"answerable": True, "refusal_reason": "partial"})
    assert not is_refusal({"answerable": True, "refusal_reason": None})


# ---------------------------------------------------------------- answers


def test_citation_validity_uses_retrieved_page_span() -> None:
    golds = [GoldSource(file_name=FILE, pages=[14])]
    ask = {
        "citations": [{"marker": "S1", "chunk_id": "c1", "file_name": FILE, "page": 13, "quote": "q"}],
        "retrieval": {"chunks": [_chunk(1, 13, page_start=13, page_end=14, chunk_id="c1")]},
    }
    assert citations_valid(ask, golds)  # the cited chunk spans pages 13-14
    ask["citations"].append(
        {"marker": "S2", "chunk_id": "x", "file_name": "other.pdf", "page": 14, "quote": "q"}
    )
    assert not citations_valid(ask, golds)  # one bad citation invalidates the answer
    assert not citations_valid({"citations": []}, golds)  # uncited answers are invalid


def _ask(answer: str, **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "answer": answer,
        "answerable": True,
        "refusal_reason": None,
        "notices": [],
        "citations": [{"marker": "S1", "chunk_id": "c1", "file_name": FILE, "page": 13, "quote": "q"}],
        "retrieval": {"chunks": [_chunk(1, 13, used_in_context=True)]},
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "latency_ms": {"total": 1200},
        "provider": {"endpoint": "nvidia", "model": "m"},
    }
    body.update(over)
    return body


def test_score_service_item_full() -> None:
    item = _item(
        type="advice_boundary",
        expected_notices=["not_advice", "referral"],
        language="roman_ur",
        answer_language="roman_ur",
    )
    ask = _ask(
        "Aap ko birth certificate chahiye, yeh general maloomat hai [S1].",
        notices=[{"code": "not_advice", "message": "m"}],
    )
    search = {"retrieval": {"chunks": [_chunk(1, 2), _chunk(2, 13)]}}
    s = score_service_item("R4", item, search, ask)
    assert s.retrieval is not None and s.retrieval["mrr@10"] == 0.5
    assert s.context_hit is True
    assert s.refused is False and s.citation_valid is True
    assert s.notice_hits == {"not_advice": True, "referral": False} and s.notices_ok is False
    assert s.detected_language == "roman_ur" and s.language_ok is True
    assert s.latency_ms == 1200 and s.input_tokens == 100
    assert failure_category(s) == "advice_boundary"


def test_failure_categories() -> None:
    un = _item(id="q_t_un", type="unanswerable_absent", answerable=False, gold_sources=[])
    refused = _ask(
        "Not in the sources.", answerable=False, refusal_reason="insufficient_context", citations=[]
    )
    assert failure_category(score_service_item("R4", un, None, refused)) == "ok"
    assert failure_category(score_service_item("R4", un, None, _ask("It takes 10 days."))) == "refusal_error"

    item = _item()
    assert failure_category(score_service_item("R4", item, None, refused)) == "false_refusal"
    miss = {"retrieval": {"chunks": [_chunk(1, 2, file_name="other.pdf")]}}
    ask_miss = _ask("x", retrieval={"chunks": [_chunk(1, 2, file_name="other.pdf", used_in_context=True)]})
    assert failure_category(score_service_item("R4", item, miss, ask_miss)) == "retrieval_miss"
    s = score_service_item("R4", item, None, None, error="ask: HTTP 500")
    assert failure_category(s) == "error"
    ok = score_service_item("R4", item, None, _ask("You need a verified birth certificate [S1]."))
    ok.judge_correctness = 0
    assert failure_category(ok) == "incorrect"
    ok.judge_correctness = 2
    assert failure_category(ok) == "ok"


def test_aggregate_and_breakdown_by_language() -> None:
    def score(lang: str, answerable: bool, refused: bool, hit: float) -> ItemScore:
        s = ItemScore("R4", f"q_{lang}_{answerable}", "single_fact", lang, lang, "test", answerable)
        s.refused = refused
        if answerable:
            s.retrieval = dict.fromkeys(
                ["hit@1", "hit@3", "hit@5", "hit@10", "recall@5", "recall@10", "mrr@10", "ndcg@10"], hit
            )
        s.latency_ms = 100.0 if lang == "en" else 300.0
        return s

    scores = [score("en", True, False, 1.0), score("en", False, True, 0), score("ur", True, True, 0.0)]
    agg = aggregate(scores)
    assert agg["n"] == 3 and agg["n_answerable"] == 2
    assert agg["hit@5"] == 0.5
    assert agg["refusal_precision"] == 0.5 and agg["refusal_recall"] == 1.0
    assert agg["false_refusal_rate"] == 0.5
    assert agg["latency_p50_ms"] == 100.0
    b = breakdown(scores)
    assert set(b["by_language"]) == {"en", "ur"}
    assert b["by_language"]["ur"]["hit@5"] == 0.0 and b["by_language"]["en"]["hit@5"] == 1.0


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("You need a verified birth certificate from the Union Council [S1].", "en"),
        ("Aap ko Union Council se marriage certificate chahiye, aur spouse ka CNIC bhi [S1].", "roman_ur"),
        ("نکاح نامہ اور شوہر کا شناختی کارڈ درکار ہے [S1]", "ur"),
        ("", "unknown"),
    ],
)
def test_detect_language(text: str, lang: str) -> None:
    assert detect_language(text) == lang
