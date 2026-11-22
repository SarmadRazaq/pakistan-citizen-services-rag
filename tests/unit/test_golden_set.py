"""Shape checks for evals/golden_set.jsonl (blueprint §10.1). The content is reviewed by the student."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, model_validator

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_set.jsonl"
SEED = Path(__file__).resolve().parents[2] / "corpus" / "SEED_SOURCES.md"


class GoldSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_name: str
    pages: list[int] | None = None
    section: str | None = None


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^q_[a-z0-9_]+$")
    question: str = Field(min_length=5)
    language: Literal["en", "ur", "roman_ur"]
    answer_language: Literal["en", "ur", "roman_ur"]
    type: Literal[
        "single_fact",
        "paraphrase",
        "multi_part",
        "contradiction",
        "ambiguity",
        "advice_boundary",
        "dated_information",
        "unanswerable_absent",
        "unanswerable_near_miss",
    ]
    answerable: bool
    gold_sources: list[GoldSource]
    agencies: list[str]
    jurisdictions: list[Literal["federal", "punjab", "sindh", "kp"]]
    expected_notices: list[Literal["not_advice", "referral", "dated_information", "emergency"]]
    source_ids: list[str]
    reference_answer: str = Field(min_length=10)
    split: Literal["dev", "test"]
    review_status: Literal["draft", "draft_needs_corpus_check", "approved", "dropped"]
    author: str
    reviewed_by: str | None
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> GoldenItem:
        if self.answerable and not self.gold_sources:
            raise ValueError("answerable items need at least one gold source")
        if not self.answerable and self.gold_sources:
            raise ValueError("unanswerable items must not have gold sources")
        if self.type.startswith("unanswerable") and self.answerable:
            raise ValueError("unanswerable types must have answerable=false")
        if self.type == "advice_boundary" and "not_advice" not in self.expected_notices:
            raise ValueError("advice_boundary items must expect the not_advice notice")
        if self.review_status == "approved" and not self.reviewed_by:
            raise ValueError("approved items must name the reviewer")
        return self


def _items() -> list[GoldenItem]:
    lines = [ln for ln in GOLDEN.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [GoldenItem.model_validate(json.loads(ln)) for ln in lines]


def test_every_item_matches_the_schema() -> None:
    assert _items()


def test_ids_are_unique() -> None:
    ids = [item.id for item in _items()]
    assert len(ids) == len(set(ids))


def test_source_ids_exist_in_seed_list() -> None:
    seed = SEED.read_text(encoding="utf-8")
    missing = {sid for item in _items() for sid in item.source_ids if f"| {sid} |" not in seed}
    assert missing == set()


@pytest.mark.parametrize("field", ["question", "reference_answer"])
def test_no_cnic_numbers_in_golden_set(field: str) -> None:
    import re

    cnic = re.compile(r"\b\d{5}-\d{7}-\d\b")
    assert not [item.id for item in _items() if cnic.search(getattr(item, field))]
