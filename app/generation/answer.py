"""Answer parsing and citation verification (blueprint §3.8, GO-1/GO-2/GO-7).

- `extract_json_object`: the first balanced `{…}` block that parses as JSON (code fences, prose
  before or after, and trailing commas are tolerated). Models do not reliably honour JSON mode.
- `parse_answer`: Pydantic validation of `{answerable, answer, citations, unanswered_parts}`.
- `verify_citations`: every citation's source must be a provided id and its quote must appear in
  that chunk (`rapidfuzz.fuzz.partial_ratio` after NFKC/whitespace normalisation ≥ quote_match_min).
  Every `[Sn]` marker in the answer needs a valid citation. Invalid citations are dropped.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rapidfuzz import fuzz

from app.guardrails.advice import has_directive_advice
from app.guardrails.output_guards import answer_language_ok, leaks_system_prompt, redact_secrets
from app.schemas import LanguageCode

MIN_QUOTE_CHARS = 8

_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_MARKER_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_MARKER = re.compile(r"S(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------- tolerant JSON


def _balanced_blocks(text: str) -> list[str]:
    """Every top-level `{…}` block, honouring strings and escapes."""
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"' and depth > 0:
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start : i + 1])
                start = -1
    return blocks


def extract_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in `text`, or None."""
    if not text:
        return None
    cleaned = _FENCE.sub("", text)
    for block in _balanced_blocks(cleaned):
        for candidate in (block, _TRAILING_COMMA.sub(r"\1", block)):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


# ---------------------------------------------------------------- answer schema


class ModelCitation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    quote: str = ""

    @field_validator("source", mode="before")
    @classmethod
    def _source(cls, value: object) -> str:
        return normalize_marker(str(value)) or str(value)

    @field_validator("quote", mode="before")
    @classmethod
    def _quote(cls, value: object) -> str:
        return "" if value is None else str(value)


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answerable: bool
    answer: str = ""
    citations: list[ModelCitation] = Field(default_factory=list)
    unanswered_parts: list[str] = Field(default_factory=list)

    @field_validator("answer", mode="before")
    @classmethod
    def _answer(cls, value: object) -> str:
        return "" if value is None else str(value)

    @field_validator("citations", mode="before")
    @classmethod
    def _citations(cls, value: object) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict) and v.get("source") is not None]
        return []

    @field_validator("unanswered_parts", mode="before")
    @classmethod
    def _parts(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        return []


def parse_answer(text: str) -> ModelAnswer | None:
    obj = extract_json_object(text)
    if obj is None:
        return None
    try:
        return ModelAnswer.model_validate(obj)
    except ValidationError:
        return None


# ---------------------------------------------------------------- citation verification


def normalize_marker(value: str) -> str | None:
    """'S1', '[S1]', 's1', ' S 1 ' → 'S1'; anything else → None."""
    match = re.fullmatch(r"\[?\s*S\s*(\d+)\s*\]?", value.strip(), re.IGNORECASE)
    return f"S{int(match.group(1))}" if match else None


_QUOTE_FOLD = str.maketrans(
    {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"}
    | {chr(0x06F0 + d): str(d) for d in range(10)}  # Urdu digits ۰-۹
    | {chr(0x0660 + d): str(d) for d in range(10)}  # Arabic-Indic digits ٠-٩
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


_TABLE_SEPARATOR_ROW = re.compile(r"^[|\-: ]+$", re.MULTILINE)


def normalize_for_quote(text: str) -> str:
    """NFKC + case folding + collapsed whitespace; typographic quotes, dashes and digits unified.

    Markdown table markup is ignored: separator rows are removed and every `|` becomes a space, so a
    quote of a table cell (words only) matches a chunk stored as a wide Markdown table.
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_FOLD)
    text = _TABLE_SEPARATOR_ROW.sub(" ", text).replace("|", " ")
    return " ".join(text.casefold().split())


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def markers_in(answer: str) -> list[str]:
    """Every `Sn` referenced by a `[Sn]` / `[Sn, Sm]` marker, in order of first use."""
    found: list[str] = []
    for group in _MARKER_GROUP.findall(answer):
        for number in _MARKER.findall(group):
            marker = f"S{int(number)}"
            if marker not in found:
                found.append(marker)
    return found


def strip_markers(answer: str, markers: set[str]) -> str:
    """Remove the given markers from the answer text (used for markers left without a citation)."""

    def repl(match: re.Match[str]) -> str:
        kept = [f"S{int(n)}" for n in _MARKER.findall(match.group(1)) if f"S{int(n)}" not in markers]
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _MARKER_GROUP.sub(repl, answer)
    return re.sub(r"[ \t]+([.,;:!?۔،])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()


@dataclass(frozen=True, slots=True)
class VerifiedCitation:
    marker: str
    quote: str
    score: float


@dataclass(frozen=True, slots=True)
class Verification:
    citations: tuple[VerifiedCitation, ...]
    dropped: tuple[str, ...] = ()  # reasons, e.g. "S9:unknown_source", "S1:quote_not_found"
    uncited_markers: tuple[str, ...] = ()  # [Sn] in the answer without a valid citation
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return bool(self.citations) and not self.uncited_markers

    @property
    def cited_markers(self) -> set[str]:
        return {c.marker for c in self.citations}


def quote_score(quote: str, chunk_text: str) -> float:
    q, c = normalize_for_quote(quote), normalize_for_quote(chunk_text)
    if not q or not c:
        return 0.0
    return float(fuzz.partial_ratio(q, c))


def verify_citations(answer: ModelAnswer, sources: Mapping[str, str], *, min_score: int) -> Verification:
    """GO-1: keep only citations whose source exists and whose quote is found in that source."""
    kept: dict[str, VerifiedCitation] = {}
    dropped: list[str] = []
    for cit in answer.citations:
        marker = normalize_marker(cit.source)
        if marker is None or marker not in sources:
            dropped.append(f"{cit.source}:unknown_source")
            continue
        quote = cit.quote.strip().strip('"').strip()
        if len(normalize_for_quote(quote)) < MIN_QUOTE_CHARS:
            dropped.append(f"{marker}:quote_too_short")
            continue
        score = quote_score(quote, sources[marker])
        if score < min_score:
            dropped.append(f"{marker}:quote_not_found")
            continue
        # Fuzzy matching forgives a changed digit or two; amounts and dates must match exactly
        # (a prefix is allowed, for a quote cut off in the middle of its last number).
        chunk_numbers = _numbers(normalize_for_quote(sources[marker]))
        quote_numbers = _numbers(normalize_for_quote(quote))
        if not all(any(c.startswith(n) for c in chunk_numbers) for n in quote_numbers):
            dropped.append(f"{marker}:quote_number_mismatch")
            continue
        best = kept.get(marker)
        if best is None or score > best.score:  # one citation per source, the best-matching quote
            kept[marker] = VerifiedCitation(marker, quote, score)
    uncited = tuple(m for m in markers_in(answer.answer) if m not in kept)
    flags: list[str] = []
    if dropped:
        flags.append("citation_dropped")
    if uncited:
        flags.append("uncited_marker")
    ordered = sorted(kept.values(), key=lambda c: int(c.marker[1:]))
    return Verification(tuple(ordered), tuple(dropped), uncited, tuple(flags))


# ---------------------------------------------------------------- full output check (GO-1..GO-7, GO-9)


class CheckKind(StrEnum):
    OK = "ok"
    MALFORMED = "malformed"  # GO-7: repair once
    LEAK = "leak"  # GO-4: block, no retry
    UNANSWERABLE = "unanswerable"  # the model said answerable=false
    CITATIONS = "citations"  # GO-1/GO-2: regenerate once
    LANGUAGE = "language"  # GO-6: regenerate once
    DIRECTIVE_ADVICE = "directive_advice"  # GO-9: regenerate once


@dataclass(frozen=True, slots=True)
class AnswerCheck:
    kind: CheckKind
    parsed: ModelAnswer | None = None
    verification: Verification | None = None
    answer: str = ""  # cleaned answer text: uncited markers stripped, secrets redacted
    unanswered_parts: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    leak: str | None = None


def evaluate_answer(
    raw: str,
    sources: Mapping[str, str],
    *,
    system_prompt: str,
    canary: str,
    answer_language: LanguageCode,
    min_score: int,
    advice_request: bool = False,
) -> AnswerCheck:
    """Parse and check one model reply. The caller decides whether to retry, accept or refuse."""
    leak = leaks_system_prompt(raw, system_prompt, canary=canary)
    if leak is not None:
        return AnswerCheck(CheckKind.LEAK, flags=(f"leak_{leak}",), leak=leak)
    parsed = parse_answer(raw)
    if parsed is None:
        return AnswerCheck(CheckKind.MALFORMED, flags=("malformed_json",))

    answer_text, answer_redacted = redact_secrets(parsed.answer.strip())
    parts: list[str] = []
    parts_redacted = False
    for part in parsed.unanswered_parts:
        clean, changed = redact_secrets(part.strip())
        parts.append(clean)
        parts_redacted = parts_redacted or changed
    flags = ["secret_redacted"] if answer_redacted or parts_redacted else []

    if not parsed.answerable or not answer_text:
        if parsed.answerable:  # "answerable": true with an empty answer is not an answer
            flags.append("empty_answer")
        return AnswerCheck(
            CheckKind.UNANSWERABLE, parsed, answer="", unanswered_parts=tuple(parts), flags=tuple(flags)
        )

    verification = verify_citations(parsed, sources, min_score=min_score)
    flags.extend(verification.flags)
    cleaned = strip_markers(answer_text, set(verification.uncited_markers))
    parts_t = tuple(parts)
    if not verification.ok:
        kind = CheckKind.CITATIONS
    elif not answer_language_ok(cleaned, answer_language):
        kind = CheckKind.LANGUAGE
        flags.append("language_mismatch")
    elif advice_request and has_directive_advice(cleaned):
        kind = CheckKind.DIRECTIVE_ADVICE
        flags.append("directive_advice")
    else:
        kind = CheckKind.OK
    return AnswerCheck(kind, parsed, verification, cleaned, parts_t, tuple(flags))
