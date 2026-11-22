"""LLM-as-a-judge (blueprint §10.4, §10.5): a rubric judge for this project's specific promises.

Scores, per answer:
- **correctness** 0–2 against the human reference answer (2 = all key facts, 1 = partly, 0 = wrong);
- **citation support** yes / no / na (do the quoted passages support the answer's claims?);
- **refusal appropriateness** (refused exactly when the corpus cannot answer);
- **advice boundary**: disclaimer present, and no personal directive ("you should file …").

The judge is `settings.judge_model` (GPT-OSS-120B by default: open-weight and a different family
from the generator, which limits self-preference). Temperature 0, JSON output, tolerant parsing,
calls cached on disk under `evals/.cache/judge/` (see `llm.py`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evals.golden import GoldenItem
from evals.llm import ChatClient, LLMError

PROMPT_VERSION = "judge-v1"

LANGUAGE_NAMES = {"en": "English", "ur": "Urdu (Arabic script)", "roman_ur": "Roman Urdu (Latin script)"}

SYSTEM = (
    "You are a strict, fair evaluator for a retrieval-augmented helpdesk about Pakistani citizen "
    "services (NADRA, passports, taxes, consumer rights, social protection). You grade ONE system "
    "answer against a reference answer written by a human expert from the official source. The answer "
    "may be in English, Urdu or Roman Urdu; judge the meaning, not the language or wording. "
    "Reply with a single JSON object and nothing else."
)

RUBRIC = """\
## Rubric
1. "correctness" (integer 0, 1 or 2), judged against the reference answer only:
   - 2: every key fact of the reference is present and nothing contradicts it.
   - 1: partly correct: some key facts are missing, or a minor detail is wrong.
   - 0: wrong, contradicts the reference, invents facts, or misses the main point.
   If the question is NOT answerable from the corpus: 2 when the system declines or clearly says the
   sources do not contain the information; 0 when it presents an answer as fact.
   If the question IS answerable and the system declined: 0.
2. "citation_support" ("yes", "no" or "na"): "yes" when the quoted passages support the claims the
   answer makes; "no" when any claim is unsupported by, or contradicts, its quote; "na" when there are
   no citations or the system declined.
3. "refusal_appropriate" (true/false): true when the system declined exactly when the question is not
   answerable from the corpus, and answered when it is answerable.
4. "disclaimer_present" (true/false): the answer or its notices say this is general information and
   not legal advice (or equivalent, in any language).
5. "personal_directive" (true/false): the answer tells the user what to do in their own personal or
   legal case (e.g. "you should withdraw your case", "file an appeal now"), rather than stating the
   general rule. Quoting the official procedure is NOT a personal directive.
6. "rationale": one short sentence explaining the correctness score.

Return exactly:
{"correctness": 0, "citation_support": "na", "refusal_appropriate": true, "disclaimer_present": false, \
"personal_directive": false, "rationale": "..."}"""


@dataclass
class JudgeInput:
    item: GoldenItem
    answer: str
    refused: bool
    refusal_reason: str | None = None
    notices: Sequence[str] = ()
    citations: Sequence[Mapping[str, Any]] = ()
    closed_book: bool = False


@dataclass
class Verdict:
    correctness: int | None
    citation_support: str | None
    refusal_appropriate: bool | None
    disclaimer_present: bool | None
    personal_directive: bool | None
    rationale: str = ""
    parsed: bool = True
    cached: bool = False

    @property
    def advice_ok(self) -> bool | None:
        if self.disclaimer_present is None or self.personal_directive is None:
            return None
        return self.disclaimer_present and not self.personal_directive


def _citation_lines(citations: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for c in citations:
        where = c.get("file_name") or "?"
        if c.get("page") is not None:
            where += f", p.{c['page']}"
        if c.get("section"):
            where += f", {c['section']}"
        quote = str(c.get("quote") or "").replace("\n", " ").strip()
        lines.append(f'[{c.get("marker", "?")}] {where}: "{quote[:600]}"')
    return "\n".join(lines) or "(none)"


def build_messages(inp: JudgeInput) -> list[dict[str, str]]:
    item = inp.item
    expected = ", ".join(item.expected_notices) or "none"
    shown = ", ".join(inp.notices) or "none"
    mode = (
        "CLOSED-BOOK: the system answered from memory without any sources (no citations expected)."
        if inp.closed_book
        else "RAG: the system answered from retrieved official documents."
    )
    asked, wanted = LANGUAGE_NAMES[item.language], LANGUAGE_NAMES[item.answer_language]
    user = f"""\
## Question ({asked}; expected answer language: {wanted})
{item.question}

## Answerable from the corpus?
{"yes" if item.answerable else "no"} (question type: {item.type})

## Reference answer (human-written)
{item.reference_answer}

## Notices the answer is expected to carry
{expected}

## System answer ({mode})
{inp.answer.strip() or "(empty)"}

## System declined to answer?
{"yes" if inp.refused else "no"}{f" (reason: {inp.refusal_reason})" if inp.refusal_reason else ""}

## Notices shown with the answer
{shown}

## Citations (marker, source, quote)
{_citation_lines(inp.citations)}

{RUBRIC}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- tolerant parsing

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    return found


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().casefold()
        if v in ("true", "yes", "y", "1"):
            return True
        if v in ("false", "no", "n", "0"):
            return False
    return None


def _to_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return max(0, min(2, round(value)))
    if isinstance(value, str):
        v = value.strip().casefold()
        words = {"correct": 2, "partial": 1, "partially correct": 1, "wrong": 0, "incorrect": 0}
        if v in words:
            return words[v]
        m = re.match(r"^\d+(?:\.\d+)?", v)
        if m:
            return max(0, min(2, round(float(m.group()))))
    return None


def _to_support(value: Any) -> str | None:
    if value is None:
        return "na"
    as_bool = _to_bool(value)
    if as_bool is not None:
        return "yes" if as_bool else "no"
    if isinstance(value, str) and value.strip().casefold() in ("na", "n/a", "none", "not applicable"):
        return "na"
    return None


def parse_verdict(text: str) -> Verdict:
    """Accepts code fences, <think> blocks, prose around the JSON and loosely typed values."""
    clean = _FENCE.sub("", _THINK.sub("", text or ""))
    for obj in reversed(_json_objects(clean)):  # the last object is the final answer
        if "correctness" not in obj:
            continue
        return Verdict(
            correctness=_to_score(obj.get("correctness")),
            citation_support=_to_support(obj.get("citation_support")),
            refusal_appropriate=_to_bool(obj.get("refusal_appropriate")),
            disclaimer_present=_to_bool(obj.get("disclaimer_present")),
            personal_directive=_to_bool(obj.get("personal_directive")),
            rationale=str(obj.get("rationale") or "")[:500],
        )
    m = re.search(r'correctness"?\s*[:=]\s*"?(\d)', clean, re.IGNORECASE)
    return Verdict(
        correctness=_to_score(m.group(1)) if m else None,
        citation_support=None,
        refusal_appropriate=None,
        disclaimer_present=None,
        personal_directive=None,
        rationale="unparsed judge output",
        parsed=False,
    )


class Judge:
    def __init__(self, client: ChatClient, model: str, *, max_tokens: int = 2048) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    async def judge(self, inp: JudgeInput) -> Verdict:
        result = await self.client.complete(
            self.model, build_messages(inp), max_tokens=self.max_tokens, json_mode=True
        )
        verdict = parse_verdict(result.content)
        verdict.cached = result.cached
        return verdict

    async def try_judge(self, inp: JudgeInput) -> tuple[Verdict | None, str | None]:
        """Never raises: a judge failure leaves the item unjudged instead of failing the run."""
        try:
            return await self.judge(inp), None
        except LLMError as exc:
            return None, str(exc)
