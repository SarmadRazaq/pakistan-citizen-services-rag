"""Parsing and validation of the X-Eval-Config header (app/core/eval_overrides.py)."""

from __future__ import annotations

import json

import pytest

from app.core.errors import AppError, ErrorCode
from app.core.eval_overrides import EvalOverrides, parse_eval_header

MODELS = ("deepseek-ai/deepseek-v3.1", "qwen/qwen3-235b-a22b")


def test_defaults() -> None:
    o = parse_eval_header("{}", MODELS)
    assert o == EvalOverrides()
    assert o.retrieval_mode == "hybrid" and o.rerank is True and o.generation_model is None
    assert o.is_default_retrieval


def test_full_header_normalised_sorted_compact() -> None:
    raw = json.dumps(
        {"rerank": False, "config": "R1", "retrieval_mode": "dense", "generation_model": MODELS[1]}
    )
    o = parse_eval_header(raw, MODELS)
    assert o.config == "R1" and o.retrieval_mode == "dense" and not o.rerank
    assert o.generation_model == MODELS[1]
    assert o.applied_json() == (
        '{"config":"R1","generation_model":"qwen/qwen3-235b-a22b","rerank":false,"retrieval_mode":"dense"}'
    )


def test_applied_json_lists_only_what_was_given_plus_nothing_else() -> None:
    assert parse_eval_header('{"rerank":false}', MODELS).applied_json() == (
        '{"rerank":false,"retrieval_mode":"hybrid"}'
    )


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[1]",
        '"x"',
        '{"surprise":1}',
        '{"retrieval_mode":"graph"}',
        '{"rerank":"no"}',
        '{"rerank":1}',
        '{"config":5}',
        '{"config":""}',
        '{"generation_model":"not/on-the-list"}',
        '{"generation_model":5}',
    ],
)
def test_bad_header_is_validation_failed(raw: str) -> None:
    with pytest.raises(AppError) as err:
        parse_eval_header(raw, MODELS)
    assert err.value.code is ErrorCode.VALIDATION_FAILED


def test_oversized_header_rejected() -> None:
    with pytest.raises(AppError):
        parse_eval_header(json.dumps({"config": "x" * 5000}), MODELS)
