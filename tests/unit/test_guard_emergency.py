"""GO-11: emergency detection, emergency Referral matching and the localized emergency Notices."""

from __future__ import annotations

import pytest

from app.generation.query_understanding import QueryUnderstanding
from app.guardrails.advice import Referral, match_emergency_referrals
from app.guardrails.emergency import is_emergency
from app.guardrails.output_guards import emergency_notices
from app.schemas import LanguageCode, NoticeCode


@pytest.mark.parametrize(
    "question",
    [
        "My husband is beating me, what can I do?",
        "My child is missing since this morning",
        "Someone is threatening to kill me",
        "my father had a heart attack",
        "میرا بچہ لاپتہ ہے",
        "مجھے جان سے مارنے کی دھمکی مل رہی ہے",
        "میرے والد کو دل کا دورہ پڑا ہے",
        "mera shohar mujhe maar raha hai",
        "meri behan ko agwa kar liya gaya",
        "mera beta gum ho gaya hai",
    ],
)
def test_emergency_questions_are_detected(question: str) -> None:
    assert is_emergency(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "How do I get an emergency passport?",
        "What is the urgent CNIC fee?",
        "My CNIC is missing, how do I get a duplicate?",
        "Lost passport report procedure",
        "What is the fee for an executive (urgent) passport?",
        "emergency travel document fee",
        "How do I file a complaint against a police officer?",
        "What is the passport fee?",
    ],
)
def test_routine_service_questions_are_not_emergencies(question: str) -> None:
    assert is_emergency(question) is False


def test_query_understanding_defaults_to_no_emergency() -> None:
    assert QueryUnderstanding.__dataclass_fields__["emergency"].default is False


def _ref(
    rid: str, *, triggers: set[str], jurisdiction: str = "federal", topics: set[str] | None = None
) -> Referral:
    return Referral(
        id=rid,
        name=f"Office {rid}",
        contact=f"c-{rid}",
        jurisdiction=jurisdiction,
        topics=frozenset(topics or set()),
        triggers=frozenset(triggers),
        source_url=f"https://example.gov.pk/{rid}",
    )


def test_match_picks_only_emergency_triggered_referrals() -> None:
    refs = [
        _ref("fire", triggers={"emergency"}),
        _ref("legal", triggers={"legal_advice"}),
        _ref("topic_only", triggers=set(), topics={"rights"}),
    ]
    got = match_emergency_referrals(refs, jurisdictions=[])
    assert [r.id for r in got] == ["fire"]


def test_match_respects_jurisdiction_and_always_includes_federal() -> None:
    refs = [
        _ref("fed", triggers={"emergency"}),
        _ref("punjab", triggers={"emergency"}, jurisdiction="punjab"),
        _ref("sindh", triggers={"emergency"}, jurisdiction="sindh"),
    ]
    got = match_emergency_referrals(refs, jurisdictions=["punjab"])
    assert {r.id for r in got} == {"fed", "punjab"}
    assert {r.id for r in match_emergency_referrals(refs, jurisdictions=[])} == {"fed"}


def test_emergency_notices_generic_first_then_one_per_referral() -> None:
    refs = [_ref("a", triggers={"emergency"}), _ref("b", triggers={"emergency"})]
    notices = emergency_notices(LanguageCode.EN, refs)
    assert len(notices) == 3
    assert all(n.code is NoticeCode.EMERGENCY for n in notices)
    assert notices[0].contact is None and notices[0].message.strip()
    assert [n.contact for n in notices[1:]] == ["c-a", "c-b"]
    assert [n.source_url for n in notices[1:]] == ["https://example.gov.pk/a", "https://example.gov.pk/b"]


def test_emergency_notices_without_referrals_still_has_the_generic_notice() -> None:
    notices = emergency_notices(LanguageCode.EN, [])
    assert len(notices) == 1
    assert notices[0].code is NoticeCode.EMERGENCY and notices[0].contact is None


def test_emergency_messages_are_localized() -> None:
    messages = {lang: emergency_notices(lang, [])[0].message for lang in LanguageCode}
    assert all(m.strip() for m in messages.values())
    assert len(set(messages.values())) == 3
