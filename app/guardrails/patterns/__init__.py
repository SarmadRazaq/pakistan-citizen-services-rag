"""Pattern lists for the guardrails (YAML next to this file) and the text normalisation they match on.

Each YAML file holds a list of `{id, lang, regex, case_sensitive?}` entries. Patterns are matched
against `normalize_for_matching(text)`; the same Urdu letter unification is applied to the regex
source, so a pattern may be written with Arabic or Urdu letter forms.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

PATTERN_DIR = Path(__file__).resolve().parent

# Arabic → Urdu letter unification (blueprint §4): Yeh, Kaf, Heh; also drop tatweel.
_URDU_UNIFY = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۀ": "ۂ", "ـ": None})
_WS = re.compile(r"[^\S\n]+")
_NL = re.compile(r"\n\s*\n+")


def unify_urdu(text: str) -> str:
    return text.translate(_URDU_UNIFY)


def normalize_for_matching(text: str, *, lower: bool = True) -> str:
    """NFKC, Urdu letter unification, collapsed spaces (newlines kept for line-start rules), case folding."""
    out = unify_urdu(unicodedata.normalize("NFKC", text))
    out = _NL.sub("\n", _WS.sub(" ", out)).strip()
    return out.casefold() if lower else out


@dataclass(frozen=True, slots=True)
class GuardPattern:
    id: str
    lang: str
    regex: re.Pattern[str]
    case_sensitive: bool = False

    def search(self, text: str) -> bool:
        """`text` is the raw (sanitised) string; normalisation happens here, per the case rule."""
        return self.regex.search(normalize_for_matching(text, lower=not self.case_sensitive)) is not None


def _compile(entry: dict[str, Any]) -> GuardPattern:
    case_sensitive = bool(entry.get("case_sensitive", False))
    source = unify_urdu(str(entry["regex"]).strip())
    flags = re.UNICODE | re.MULTILINE
    if not case_sensitive:
        flags |= re.IGNORECASE
    return GuardPattern(
        id=str(entry["id"]),
        lang=str(entry.get("lang", "any")),
        regex=re.compile(source, flags),
        case_sensitive=case_sensitive,
    )


@cache
def _load(name: str) -> dict[str, Any]:
    data = yaml.safe_load((PATTERN_DIR / name).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"pattern file {name} must be a mapping")
    return data


@cache
def load_patterns(name: str, section: str = "patterns") -> tuple[GuardPattern, ...]:
    """Compile one section of a pattern file (cached for the process)."""
    entries = _load(name).get(section) or []
    return tuple(_compile(e) for e in entries)


def first_match(patterns: tuple[GuardPattern, ...], text: str) -> GuardPattern | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern
    return None
