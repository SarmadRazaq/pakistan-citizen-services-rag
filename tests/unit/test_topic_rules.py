"""Closed Topic list rules: slug normalisation and sync with the frontend options (GLOSSARY.md Topic)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.ingestion.metadata import TOPICS, normalize_topic

TOPICS_TS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "topics.ts"
VALUE_RE = re.compile(r"value:\s*[\"']([^\"']+)[\"']")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Social Protection ", "social_protection"),
        ("Driving-Licence", "driving_licence"),
        ("PASSPORT", "passport"),
        ("consumer   protection", "consumer_protection"),
        ("driving - licence", "driving_licence"),
        ("tax", "tax"),
        ("cnic", "cnic"),  # normalising does not validate
    ],
)
def test_normalize_topic(raw: str, expected: str) -> None:
    assert normalize_topic(raw) == expected


def test_every_topic_slug_is_already_normalised() -> None:
    assert all(normalize_topic(t) == t for t in TOPICS)


def test_frontend_topic_options_match_backend_topics() -> None:
    values = VALUE_RE.findall(TOPICS_TS.read_text(encoding="utf-8"))
    assert len(values) == len(set(values))
    assert set(values) == set(TOPICS)
