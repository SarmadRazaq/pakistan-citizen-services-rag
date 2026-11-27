"""Language detection in code, province detection and the tolerant helper call (blueprint §4)."""

from __future__ import annotations

import pytest

from app.core.errors import AppError, ErrorCode
from app.generation.query_understanding import (
    canonical_agency,
    detect_jurisdictions,
    detect_language,
    understand,
)
from app.guardrails.advice import is_advice_request
from app.schemas import Jurisdiction, LanguageCode
from tests.fakes import FakeLLM
from tests.test_ask_support import helper_reply

EN, UR, RU = LanguageCode.EN, LanguageCode.UR, LanguageCode.ROMAN_UR


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("What is the fee for an urgent passport?", EN),
        ("How do I renew my CNIC?", EN),
        ("ارجنٹ پاسپورٹ کی فیس کتنی ہے؟", UR),
        ("NADRA کا شناختی کارڈ کیسے بنتا ہے؟", UR),
        ("Urgent passport banwane ki fee kitni hai?", RU),
        ("CNIC renew karwane ka tareeqa kya hai?", RU),
        ("Passport renewal ka process kya hai?", RU),  # code-switched, Roman Urdu dominant (E23)
        ("What is the fee, kitni hai?", EN),  # code-switched, English dominant
        ("kia pasport ki fees barh gayi hai", RU),  # non-standard spelling (E24)
        ("NADRA", EN),
    ],
)
def test_detect_language(text: str, lang: LanguageCode) -> None:
    assert detect_language(text) is lang


@pytest.mark.parametrize(
    ("text", "places"),
    [
        ("Driving licence in Lahore", (Jurisdiction.PUNJAB,)),
        ("Karachi mein domicile kaise banwaun?", (Jurisdiction.SINDH,)),
        ("Sehat card KP mein", (Jurisdiction.KP,)),
        ("پشاور میں ڈرائیونگ لائسنس", (Jurisdiction.KP,)),
        ("لاہور اور کراچی", (Jurisdiction.PUNJAB, Jurisdiction.SINDH)),
        ("How do I apply for a passport?", ()),
        ("Mera kya kasoor hai?", ()),
    ],
)
def test_detect_jurisdictions(text: str, places: tuple[Jurisdiction, ...]) -> None:
    assert detect_jurisdictions(text) == places


def test_canonical_agency() -> None:
    assert canonical_agency("nadra") == "NADRA"
    assert canonical_agency("Federal Board of Revenue") == "FBR"
    assert canonical_agency("Some Unknown Office") is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("mera case court mein hai, ab kya karun?", True),
        ("Should I file an appeal against the tax notice?", True),
        ("میرا مقدمہ عدالت میں ہے، اب کیا کروں؟", True),
        ("What is the fee for an urgent passport?", False),
        ("What can I do if I lose my CNIC?", False),
    ],
)
def test_advice_request_patterns(question: str, expected: bool) -> None:
    assert is_advice_request(question) is expected


async def test_understand_parses_the_helper_json() -> None:
    llm = FakeLLM(
        replies=[
            helper_reply(
                english_query="What is the urgent passport fee?",
                urdu_query="ارجنٹ پاسپورٹ کی فیس کتنی ہے؟",
                sub_queries=["a", "b", "c", "d"],
                agencies=["nadra", "Unknown Office"],
                jurisdictions=["Punjab", "mars"],
                advice_request="true",
                injection_suspect=True,
            )
        ]
    )
    q = await understand("Urgent passport ki fee kitni hai?", llm)
    assert q.helper_ok and q.language is RU
    assert q.sub_queries == ("a", "b", "c")  # capped at 3
    assert q.agencies == ("NADRA",)
    assert q.jurisdictions == (Jurisdiction.PUNJAB,)
    assert q.advice_request and q.injection_suspect and "injection_suspect" in q.flags
    assert q.variants[:3] == (
        "Urgent passport ki fee kitni hai?",
        "What is the urgent passport fee?",
        "ارجنٹ پاسپورٹ کی فیس کتنی ہے؟",
    )
    assert llm.seen[0][0].content.startswith("You analyse one question")


@pytest.mark.parametrize(
    "reply",
    [RuntimeError("boom"), AppError(ErrorCode.LLM_UNAVAILABLE), "garbage", '{"advice_request": [1, 2'],
)
async def test_understand_degrades_on_any_failure(reply: object) -> None:
    q = await understand("Lahore mein CNIC fee kitni hai?", FakeLLM(replies=[reply]))  # type: ignore[list-item]
    assert not q.helper_ok
    assert q.variants == ("Lahore mein CNIC fee kitni hai?",)
    assert q.language is RU
    assert q.jurisdictions == (Jurisdiction.PUNJAB,)  # still detected in code
    assert q.agencies == ()


async def test_understand_without_llm() -> None:
    q = await understand("NADRA CNIC renewal fee?", None)
    assert q.variants == ("NADRA CNIC renewal fee?",) and q.agencies == ("NADRA",)
