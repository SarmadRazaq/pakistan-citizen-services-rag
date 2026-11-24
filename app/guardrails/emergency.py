"""GO-11 emergency detection: does the question describe someone in danger now (code patterns)."""

from __future__ import annotations

from app.guardrails.patterns import first_match, load_patterns

EMERGENCY_FILE = "emergency.yaml"


def is_emergency(question: str) -> bool:
    return first_match(load_patterns(EMERGENCY_FILE, "patterns"), question) is not None
