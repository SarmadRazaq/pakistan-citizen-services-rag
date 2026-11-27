"""Ingest-time injection scoring for chunks (GD-10).

The real classifier (Prompt-Guard, `InjectionClassifier`) is injected into the ingestion service when
it is available. Without it, this small pattern heuristic still flags the obvious cases ("ignore all
previous instructions", "SYSTEM:", the Urdu equivalents), so the ⚠ flag works offline. A flagged chunk
is stored, not dropped: it may be legitimate text quoting such phrases, and it is always passed to the
model as data.
"""

from __future__ import annotations

import re

INJECTION_FLAG_THRESHOLD = 0.5

_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all|earlier)\b.{0,20}"
            r"\b(instructions?|rules|prompts?)\b",
            re.I | re.S,
        ),
        0.9,
    ),
    (re.compile(r"(^|\n)\s*(system|assistant)\s*:", re.I), 0.6),
    (re.compile(r"\byou\s+are\s+now\b|\bact\s+as\b.{0,40}\b(unfiltered|jailbroken|dan)\b", re.I), 0.7),
    (re.compile(r"\b(reveal|print|show)\b.{0,30}\b(system\s+prompt|hidden\s+instructions)\b", re.I), 0.8),
    (re.compile(r"\breply\s+only\b|\brespond\s+only\s+with\b", re.I), 0.5),
    (re.compile(r"(پچھلی|سابقہ|تمام)\s*(ہدایات|ہدایت)\s*(کو)?\s*(نظر\s*انداز|بھول)"), 0.9),
)  # fmt: skip


def heuristic_injection_score(text: str) -> float:
    score = 0.0
    for pattern, weight in _PATTERNS:
        if pattern.search(text):
            score = max(score, weight)
    return score
