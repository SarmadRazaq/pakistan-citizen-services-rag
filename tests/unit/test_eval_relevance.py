"""Chunk-to-gold relevance matching (evals/relevance.py, blueprint §10.1)."""

from __future__ import annotations

import pytest
from evals.golden import GoldSource
from evals.relevance import chunk_grade, chunk_pages, file_matches, grade, is_relevant, split_source_id

MANUAL = "ID-05__nadra-registration-policy-6-0-2-en.pdf"
PLAIN = "nadra-registration-policy-6-0-2-en.pdf"


@pytest.mark.parametrize(
    ("chunk_file", "gold_file", "expected"),
    [
        (MANUAL, MANUAL, True),
        (MANUAL, PLAIN, True),  # manual-save name vs the plain gold name
        (PLAIN, MANUAL, True),
        ("id-05__NADRA-Registration-Policy-6-0-2-EN.pdf", MANUAL, True),  # case-insensitive
        ("ID-05__renamed-slug.pdf", MANUAL, True),  # same source id, different slug
        ("ID-06__nadra-registration-policy-6-0-2-ur.pdf", MANUAL, False),  # the Urdu twin is another doc
        ("ID-02a__nadra-cnic-new.html", "ID-02b__nadra-cnic-renew.html", False),
        ("storage/u_public/" + MANUAL, MANUAL, True),  # a path, not just a name
        ("other.pdf", PLAIN, False),
        (None, PLAIN, False),
    ],
)
def test_file_matches(chunk_file: str | None, gold_file: str, expected: bool) -> None:
    assert file_matches(chunk_file, gold_file) is expected


def test_split_source_id() -> None:
    assert split_source_id(MANUAL) == ("id-05", PLAIN)
    assert split_source_id("ID-02a__nadra-cnic-new.html") == ("id-02a", "nadra-cnic-new.html")
    assert split_source_id(PLAIN) == (None, PLAIN)


def test_page_span_overlap() -> None:
    gold = GoldSource(file_name=MANUAL, pages=[14])
    assert grade({"file_name": MANUAL, "page": 14}, gold) == 2
    assert grade({"file_name": MANUAL, "page": 13}, gold) == 1  # right file, wrong page
    assert grade({"file_name": MANUAL, "page_start": 13, "page_end": 15}, gold) == 2
    assert grade({"file_name": "x.pdf", "page": 14}, gold) == 0
    assert grade({"file_name": MANUAL}, gold) == 1  # page unknown: cannot confirm the place
    assert chunk_pages({"page_start": 3, "page_end": 1}) == {3}
    assert chunk_pages({"pages": [2, "4"]}) == {2, 4}


def test_section_match_for_html() -> None:
    html = "ID-02b__nadra-cnic-renew.html"
    gold = GoldSource(file_name=html, section="Required documents")
    assert grade({"file_name": html, "section_path": ["CNIC Renewal", "Required Documents"]}, gold) == 2
    assert grade({"file_name": html, "section": "CNIC Renewal › Required documents"}, gold) == 2
    assert grade({"file_name": html, "section_path": ["CNIC Renewal", "Fees"]}, gold) == 1
    # no section info on the chunk: the heading in the text counts (contextual chunk headers)
    assert grade({"file_name": html, "text": "Required documents: old CNIC, ..."}, gold) == 2
    assert grade({"file_name": html, "text": "Fee schedule"}, gold) == 1
    nested = GoldSource(file_name=html, section="CNIC Renewal > Required documents")
    assert (
        grade({"file_name": html, "section_path": ["CNIC Renewal", "Required documents", "Overseas"]}, nested)
        == 2
    )


def test_whole_document_gold_and_multiple_golds() -> None:
    whole = GoldSource(file_name=PLAIN)
    assert grade({"file_name": MANUAL, "page": 40}, whole) == 2
    golds = [GoldSource(file_name="a.pdf", pages=[1]), GoldSource(file_name=MANUAL, pages=[13])]
    chunk = {"file_name": MANUAL, "page": 13}
    assert chunk_grade(chunk, golds) == 2 and is_relevant(chunk, golds)
    assert not is_relevant({"file_name": MANUAL, "page": 2}, golds)
