"""The golden-set item schema (blueprint §10.1): the single source of truth.

`tests/unit/test_golden_set.py` imports these models, so the shape checks and the harness can never
disagree about what an item looks like.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.jsonl"

Language = Literal["en", "ur", "roman_ur"]
ItemType = Literal[
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
NoticeCode = Literal["not_advice", "referral", "dated_information", "emergency"]
ReviewStatus = Literal["draft", "draft_needs_corpus_check", "approved", "dropped"]
Split = Literal["dev", "test"]


class GoldSource(BaseModel):
    """A gold label at document + page/section level (never chunk ids, so it survives re-chunking)."""

    model_config = ConfigDict(extra="forbid")
    file_name: str
    pages: list[int] | None = None
    section: str | None = None


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^q_[a-z0-9_]+$")
    question: str = Field(min_length=5)
    language: Language
    answer_language: Language
    type: ItemType
    answerable: bool
    gold_sources: list[GoldSource]
    agencies: list[str]
    jurisdictions: list[Literal["federal", "punjab", "sindh", "kp"]]
    expected_notices: list[NoticeCode]
    source_ids: list[str]
    reference_answer: str = Field(min_length=10)
    split: Split
    review_status: ReviewStatus
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

    @property
    def is_multilingual(self) -> bool:
        return self.language != "en"


def read_items(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    """Every item in the file, whatever its review status (validated)."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [GoldenItem.model_validate(json.loads(ln)) for ln in lines]


def load_golden(
    path: Path = GOLDEN_PATH,
    include_drafts: bool = False,
    split: Literal["dev", "test", "all"] = "all",
) -> list[GoldenItem]:
    """Items to evaluate: `approved` only, unless drafts are included. `dropped` items never run."""
    items = read_items(path)
    if include_drafts:
        items = [i for i in items if i.review_status != "dropped"]
    else:
        items = [i for i in items if i.review_status == "approved"]
    if split != "all":
        items = [i for i in items if i.split == split]
    return items


def golden_sha256(path: Path = GOLDEN_PATH) -> str:
    """Recorded in metrics.json, so every published number names the exact golden set it used."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
