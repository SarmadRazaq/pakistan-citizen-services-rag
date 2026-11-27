"""Shape checks for evals/golden_set.jsonl (blueprint §10.1). The content is reviewed by the student.

The item schema lives in `evals/golden.py`: one source of truth for these checks and the harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.golden import GoldenItem

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_set.jsonl"
SEED = Path(__file__).resolve().parents[2] / "corpus" / "SEED_SOURCES.md"


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

    def listed(sid: str) -> bool:
        # A seed row may cover several pages saved as sub-sources (TX-02 -> TX-02a, TX-02b, ...).
        return f"| {sid} |" in seed or f"| {sid.rstrip('abcdefghijklmnopqrstuvwxyz')} |" in seed

    missing = {sid for item in _items() for sid in item.source_ids if not listed(sid)}
    assert missing == set()


@pytest.mark.parametrize("field", ["question", "reference_answer"])
def test_no_cnic_numbers_in_golden_set(field: str) -> None:
    import re

    cnic = re.compile(r"\b\d{5}-\d{7}-\d\b")
    assert not [item.id for item in _items() if cnic.search(getattr(item, field))]
