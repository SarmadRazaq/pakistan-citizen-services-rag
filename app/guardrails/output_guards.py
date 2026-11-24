"""Output guardrail primitives (blueprint §7.2): leakage, secrets, language, and the code-added notices.

The orchestration (what to retry, when to refuse) lives in `app/generation/answer.py` and the graph.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from app.core.logging import REDACTED, redact_text
from app.generation.prompts import message
from app.guardrails.advice import Referral
from app.schemas import LanguageCode, Notice, NoticeCode

LEAK_NGRAM = 8
DATED_DOC_TYPES = frozenset({"fee_schedule", "notification", "deadline"})
_WORD = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
_KEY_LINE = re.compile(
    r"(?im)\b[A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*\s*[=:]\s*['\"]?[^\s'\"]{8,}"
)
_MARKERS = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]", re.IGNORECASE)
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|«[^»]*»")
_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")


def script_ratio(text: str) -> tuple[float, float]:
    """(arabic share, latin share) of the letters in `text`; (0, 0) when there are no letters."""
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    total = arabic + latin
    return (arabic / total, latin / total) if total else (0.0, 0.0)


# ---------------------------------------------------------------- GO-4 system prompt leakage


def _words(text: str) -> list[str]:
    return [w.casefold() for w in _WORD.findall(text)]


def ngrams(text: str, n: int = LEAK_NGRAM) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def leaks_system_prompt(text: str, system_prompt: str, *, canary: str, n: int = LEAK_NGRAM) -> str | None:
    """'canary' or 'ngram_overlap' when `text` exposes the system prompt, else None."""
    if canary and canary.casefold() in text.casefold():
        return "canary"
    if ngrams(text, n) & ngrams(system_prompt, n):
        return "ngram_overlap"
    return None


# ---------------------------------------------------------------- GO-5 secrets


def redact_secrets(text: str) -> tuple[str, bool]:
    """Redact key formats (shared with log redaction) and `KEY=value` lines."""
    redacted = _KEY_LINE.sub(REDACTED, redact_text(text))
    return redacted, redacted != text


# ---------------------------------------------------------------- GO-6 language / script


def answer_language_ok(answer: str, expected: LanguageCode) -> bool:
    """Simple script check: Urdu answers must be mostly Arabic script; en / roman_ur mostly Latin.

    Markers and quoted passages are ignored (quotes stay in the source language by design).
    """
    body = _QUOTED.sub(" ", _MARKERS.sub(" ", answer))
    arabic, latin = script_ratio(body)
    if arabic == 0 and latin == 0:
        return True
    if expected is LanguageCode.UR:
        return arabic >= 0.5
    return latin >= 0.6


# ---------------------------------------------------------------- notices (GO-9, GO-10, privacy)


def advice_notices(language: LanguageCode, referrals: Sequence[Referral]) -> list[Notice]:
    """`not_advice` always; one `referral` per matched *verified* entry (never model-generated)."""
    notices = [Notice(code=NoticeCode.NOT_ADVICE, message=message("not_advice", language))]
    for ref in referrals:
        notices.append(
            Notice(
                code=NoticeCode.REFERRAL,
                message=message("referral", language, name=ref.name),
                contact=ref.contact,
                source_url=ref.source_url,
            )
        )
    return notices


def emergency_notices(language: LanguageCode, referrals: Sequence[Referral]) -> list[Notice]:
    """A generic emergency Notice, then one per matched *verified* emergency Referral."""
    notices = [Notice(code=NoticeCode.EMERGENCY, message=message("emergency", language))]
    for ref in referrals:
        notices.append(
            Notice(
                code=NoticeCode.EMERGENCY,
                message=message("emergency_referral", language, name=ref.name),
                contact=ref.contact,
                source_url=ref.source_url,
            )
        )
    return notices


def add_months(day: date, months: int) -> date:
    month_index = day.month - 1 + months
    year, month = day.year + month_index // 12, month_index % 12 + 1
    for candidate_day in (day.day, 30, 29, 28):
        try:
            return day.replace(year=year, month=month, day=candidate_day)
        except ValueError:
            continue
    raise ValueError("unreachable")


@dataclass(frozen=True, slots=True)
class DatedSource:
    doc_id: str
    doc_type: str | None
    effective_date: date | None
    file_name: str
    source_url: str | None


def dated_notices(
    sources: Iterable[DatedSource], *, today: date, stale_after_months: int, language: LanguageCode
) -> list[Notice]:
    """GO-10: one notice per cited fee/notification/deadline document that is stale or undated."""
    cutoff = add_months(today, -stale_after_months)
    notices: list[Notice] = []
    seen: set[str] = set()
    for src in sources:
        if src.doc_id in seen or (src.doc_type or "").lower() not in DATED_DOC_TYPES:
            continue
        seen.add(src.doc_id)
        if src.effective_date is None:
            text = message("undated_information", language, file=src.file_name)
        elif src.effective_date < cutoff:
            text = message(
                "dated_information", language, file=src.file_name, date=src.effective_date.isoformat()
            )
        else:
            continue
        notices.append(Notice(code=NoticeCode.DATED_INFORMATION, message=text, source_url=src.source_url))
    return notices


def third_party_notice(language: LanguageCode) -> Notice:
    return Notice(code=NoticeCode.THIRD_PARTY_PROCESSING, message=message("third_party_processing", language))
