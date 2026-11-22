"""Query understanding (blueprint §4, §6 `understand`): language in code, then one helper LLM call.

Language detection never depends on the model:
- Arabic script dominates → `ur`;
- Latin script with more Roman Urdu lexicon hits than English function words → `roman_ur`;
- otherwise `en`. Code-switched text gets its dominant language (E23).

The helper call (`purpose="helper"`) translates, transliterates, decomposes and flags the question.
Any failure (exception, timeout, unparsable JSON) degrades to `variants=(original,)` and the
pipeline keeps answering. Provinces and cities are also detected in code, so the jurisdiction filter
works even when the helper is down.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.errors import AppError
from app.generation.answer import extract_json_object
from app.generation.prompts import build_helper_messages
from app.guardrails.advice import is_advice_request
from app.guardrails.patterns import normalize_for_matching
from app.protocols import LLMClient, LLMResult
from app.schemas import Jurisdiction, LanguageCode

log = structlog.get_logger(__name__)

HELPER_MAX_TOKENS = 400
MAX_SUB_QUERIES = 3

# ---------------------------------------------------------------- language detection

_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")
_LATIN_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Words that are common in Roman Urdu and rare as English words. Ambiguous ones ("to", "me", "is",
# "in", "par", "na") are left out so English questions are not mislabelled.
ROMAN_URDU_LEXICON = frozenset(
    """
    kya kia kiya hai hain hay hy ha ho hoga hogi honge hota hoti hotay hote tha thi thay ki ka ke ko
    se mein main mai mujhe mujhy mera meri mere meray hum hamara hamari humara aap apna apni apne
    apnay tum tumhara tumhari unka unki unke iska iski iske uska uski uske yeh ye woh wo kitni kitna
    kitne kitnay kaise kaisay kese kesay kahan kahaan kab kyun kyon kyu kaun kon konsa konsi kis
    kisi kuch koi sab bohat bahut zyada kam karna karni karne karen karein karo karun karoon karwana
    karwani karwaun karwaon karwain banwana banwani banwane banwaun banwain bana banana banane
    chahiye chahye chaiye chahta chahti sakta sakti sakte milta milti milega milegi lagta lagti
    lagega lagegi lena leni lene dena deni dene jana jaana jani jaye jayega batao bataen batain
    bataein bataiye btao dikhao liye liay wala wali walay wale baad pehle phir agar lekin magar aur
    ya nahi nahin nai nhi jee ji haan han acha theek sahi zaroori zaruri tareeqa tareeka tariqa kaam
    paisa paise raqam daftar shanakhti shnakhti qanoon huqooq haq
    """.split()  # noqa: SIM905 (a word list reads better than a quoted literal)
)
ENGLISH_FUNCTION_WORDS = frozenset(
    """
    the a an is are was were be been am what which who whom whose how when where why do does did can
    could should would will shall may might must of for to from with without about into on at by and
    or but if then than this that these those my your our their it its i you we they he she there
    here have has had not no any some all each every get apply need required much many
    """.split()  # noqa: SIM905 (a word list reads better than a quoted literal)
)


def detect_language(text: str) -> LanguageCode:
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    if arabic and arabic >= latin:
        return LanguageCode.UR
    words = _LATIN_WORD.findall(text.lower())
    roman_hits = sum(1 for w in words if w in ROMAN_URDU_LEXICON)
    english_hits = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    if roman_hits > english_hits and roman_hits >= 1:
        return LanguageCode.ROMAN_UR
    return LanguageCode.EN


# ---------------------------------------------------------------- provinces and agencies in code

PROVINCE_NAMES: dict[Jurisdiction, tuple[str, ...]] = {
    Jurisdiction.PUNJAB: (
        "punjab",
        "panjab",
        "lahore",
        "rawalpindi",
        "pindi",
        "faisalabad",
        "multan",
        "gujranwala",
        "sialkot",
        "bahawalpur",
        "sargodha",
        "sheikhupura",
        "gujrat",
        "jhelum",
        "sahiwal",
        "okara",
        "rahim yar khan",
        "dera ghazi khan",
        "chakwal",
        "attock",
        "murree",
        "پنجاب",
        "لاہور",
        "راولپنڈی",
        "فیصل آباد",
        "ملتان",
        "گوجرانوالہ",
        "سیالکوٹ",
        "بہاولپور",
        "سرگودھا",
    ),
    Jurisdiction.SINDH: (
        "sindh",
        "sind",
        "karachi",
        "hyderabad",
        "sukkur",
        "larkana",
        "mirpurkhas",
        "mirpur khas",
        "nawabshah",
        "shaheed benazirabad",
        "thatta",
        "badin",
        "jacobabad",
        "khairpur",
        "سندھ",
        "کراچی",
        "حیدرآباد",
        "حیدر آباد",
        "سکھر",
        "لاڑکانہ",
        "میرپورخاص",
        "نوابشاہ",
    ),
    Jurisdiction.KP: (
        "kp",
        "kpk",
        "khyber pakhtunkhwa",
        "khyber-pakhtunkhwa",
        "khyber pakhtoonkhwa",
        "nwfp",
        "peshawar",
        "mardan",
        "abbottabad",
        "swat",
        "mingora",
        "kohat",
        "bannu",
        "dera ismail khan",
        "d.i. khan",
        "di khan",
        "charsadda",
        "nowshera",
        "mansehra",
        "swabi",
        "chitral",
        "خیبر پختونخوا",
        "خیبرپختونخوا",
        "پختونخوا",
        "پشاور",
        "مردان",
        "ایبٹ آباد",
        "سوات",
        "کوہاٹ",
        "بنوں",
    ),
}

AGENCY_ALIASES: dict[str, tuple[str, ...]] = {
    "NADRA": ("nadra", "نادرا"),
    "DGIP": ("dgip", "immigration and passports", "immigration & passports"),
    "FBR": ("fbr", "federal board of revenue", "ایف بی آر"),
    "BISP": ("bisp", "benazir income support"),
    "MOHTASIB": ("mohtasib", "ombudsman", "محتسب"),
}


def _contains_name(haystack: str, name: str) -> bool:
    if name.isascii():
        return re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", haystack) is not None
    return name in haystack


def detect_jurisdictions(text: str) -> tuple[Jurisdiction, ...]:
    norm = normalize_for_matching(text)
    found = [j for j, names in PROVINCE_NAMES.items() if any(_contains_name(norm, n) for n in names)]
    return tuple(found)


def canonical_agency(name: str) -> str | None:
    norm = normalize_for_matching(name)
    if not norm:
        return None
    for canonical, aliases in AGENCY_ALIASES.items():
        if norm == canonical.casefold() or any(_contains_name(norm, a) for a in aliases):
            return canonical
    return None


def detect_agencies(text: str) -> tuple[str, ...]:
    norm = normalize_for_matching(text)
    return tuple(c for c, aliases in AGENCY_ALIASES.items() if any(_contains_name(norm, a) for a in aliases))


# ---------------------------------------------------------------- helper call


class HelperOutput(BaseModel):
    """What the helper model returns. Lenient: unknown keys ignored, bad values coerced or dropped."""

    model_config = ConfigDict(extra="ignore")

    english_query: str | None = None
    urdu_query: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    agencies: list[str] = Field(default_factory=list)
    jurisdictions: list[str] = Field(default_factory=list)
    advice_request: bool = False
    off_topic: bool = False
    injection_suspect: bool = False

    @field_validator("sub_queries", "agencies", "jurisdictions", mode="before")
    @classmethod
    def _as_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(v) for v in value if isinstance(v, str | int | float) and str(v).strip()]
        return []

    @field_validator("english_query", "urdu_query", mode="before")
    @classmethod
    def _as_text(cls, value: object) -> str | None:
        return value.strip() or None if isinstance(value, str) else None

    @field_validator("advice_request", "off_topic", "injection_suspect", mode="before")
    @classmethod
    def _as_bool(cls, value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)


@dataclass(frozen=True, slots=True)
class QueryUnderstanding:
    original: str
    language: LanguageCode
    english_query: str | None = None
    urdu_query: str | None = None
    sub_queries: tuple[str, ...] = ()
    agencies: tuple[str, ...] = ()
    jurisdictions: tuple[Jurisdiction, ...] = ()
    advice_request: bool = False
    off_topic: bool = False
    injection_suspect: bool = False
    helper_ok: bool = False
    flags: tuple[str, ...] = ()
    usage: tuple[int, int] = (0, 0)  # helper input/output tokens
    variants: tuple[str, ...] = field(default_factory=tuple)


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = normalize_for_matching(item)
        if item and key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return tuple(out)


def build_variants(
    original: str, english: str | None, urdu: str | None, sub_queries: tuple[str, ...]
) -> tuple[str, ...]:
    """Original + English + Urdu-script variants, plus sub-queries when the question was decomposed."""
    extra = sub_queries if len(sub_queries) > 1 else ()
    return _dedupe([original, english or "", urdu or "", *extra])


def _parse_helper(text: str) -> HelperOutput | None:
    obj = extract_json_object(text)
    if obj is None:
        return None
    try:
        return HelperOutput.model_validate(obj)
    except ValidationError:
        return None


async def understand(question: str, llm: LLMClient | None) -> QueryUnderstanding:
    """Language + provinces in code, then the helper call. Never raises for a helper failure."""
    language = detect_language(question)
    code_jurisdictions = detect_jurisdictions(question)
    code_agencies = detect_agencies(question)
    code_advice = is_advice_request(question)
    degraded = QueryUnderstanding(
        original=question,
        language=language,
        agencies=code_agencies,
        jurisdictions=code_jurisdictions,
        advice_request=code_advice,
        variants=(question,),
        flags=("understand_degraded",),
    )
    if llm is None:
        return degraded

    try:
        result: LLMResult = await llm.complete(
            build_helper_messages(question), purpose="helper", max_tokens=HELPER_MAX_TOKENS
        )
    except AppError as exc:
        log.warning("understand.helper_failed", code=exc.code.slug)
        return degraded
    except Exception as exc:  # the helper is best-effort: any failure degrades, never fails the ask
        log.warning("understand.helper_failed", error=type(exc).__name__)
        return degraded

    parsed = _parse_helper(result.text)
    usage = (result.input_tokens, result.output_tokens)
    if parsed is None:
        log.warning("understand.helper_unparsable")
        return replace(degraded, usage=usage)

    sub_queries = _dedupe(parsed.sub_queries)[:MAX_SUB_QUERIES]
    model_jurisdictions: list[Jurisdiction] = []
    for value in parsed.jurisdictions:
        try:
            model_jurisdictions.append(Jurisdiction(value.strip().lower()))
        except ValueError:
            continue
    jurisdictions = tuple(dict.fromkeys([*code_jurisdictions, *model_jurisdictions]))
    model_agencies = [a for a in (canonical_agency(x) for x in parsed.agencies) if a]
    agencies = tuple(dict.fromkeys([*code_agencies, *model_agencies]))
    flags = ("injection_suspect",) if parsed.injection_suspect else ()
    return QueryUnderstanding(
        original=question,
        language=language,
        english_query=parsed.english_query,
        urdu_query=parsed.urdu_query,
        sub_queries=sub_queries,
        agencies=agencies,
        jurisdictions=jurisdictions,
        advice_request=parsed.advice_request or code_advice,
        off_topic=parsed.off_topic,
        injection_suspect=parsed.injection_suspect,
        helper_ok=True,
        flags=flags,
        usage=usage,
        variants=build_variants(question, parsed.english_query, parsed.urdu_query, sub_queries),
    )
