"""Per-page text-quality check (blueprint §3.2, E46): "text that exists but is junk".

Real failures seen in the corpus, all of which *look* like successful extraction:
- a garbage Latin OCR layer over an Urdu scan (NA Urdu Constitution): Latin script, but almost no
  real English words → low common-word hit rate;
- scrambled Urdu letter order (NADRA Urdu Registration Policy, Punjab Urdu Act): "میں" comes out
  as "ںیم" and "کے" as "ےک". Words then *start* with letters that never start an Urdu word
  (ں, ے, ھ, …), which the scramble detector counts. The common-word hit rate is 0.01–0.03 there,
  against about 0.25 for good English;
- no text at all (scans, vector-outline text) → too short.

A failing page is OCR'd from its rendered image when Tesseract is available (see `ocr.py`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.ingestion.language import ARABIC_LETTER, LATIN_LETTER, normalize_urdu, strip_diacritics

MIN_PAGE_CHARS = 25
MIN_WORDS_FOR_DICT_CHECK = 20  # short pages (titles, signatures) are not judged on vocabulary
MIN_ARABIC_WORDS_FOR_SCRAMBLE_CHECK = 10
JUNK_REASONS = frozenset({"wrong_script", "scrambled_urdu", "low_dict_hit"})
URDU_SCRAMBLE_MAX = 0.02  # well-formed Urdu ≈ 0; measured 0.03–0.15 on the scrambled NADRA Urdu PDF

URDU_COMMON = frozenset(
    [
        "کے",
        "کی",
        "کا",
        "میں",
        "ہے",
        "اور",
        "سے",
        "کو",
        "پر",
        "یہ",
        "ہیں",
        "کہ",
        "نہیں",
        "کیا",
        "جو",
        "گا",
        "گی",
        "ایک",
        "لیے",
        "بھی",
        "شناختی",
        "کارڈ",
        "درخواست",
        "فیس",
        "حکومت",
        "قانون",
        "ہو",
        "وہ",
        "اس",
        "ان",
        "جس",
        "جب",
        "تک",
        "کر",
        "کیے",
        "گئے",
        "گیا",
        "والے",
        "والی",
        "کسی",
        "کوئی",
        "دیا",
        "جائے",
        "جاتا",
        "ہوں",
        "یا",
        "مطابق",
        "صورت",
        "شہری",
        "پاکستان",
        "نادرا",
        "تاریخ",
        "ادارہ",
        "سرٹیفکیٹ",
        "دفتر",
        "شکایت",
    ]
)
ENGLISH_COMMON = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "for",
        "be",
        "by",
        "or",
        "with",
        "on",
        "as",
        "shall",
        "any",
        "not",
        "fee",
        "card",
        "application",
        "act",
        "section",
        "government",
        "rules",
        "a",
        "an",
        "are",
        "at",
        "from",
        "this",
        "that",
        "which",
        "it",
        "may",
        "will",
        "has",
        "have",
        "under",
        "such",
        "all",
        "other",
        "their",
        "its",
        "was",
        "been",
        "if",
        "no",
        "person",
        "date",
        "issue",
        "registration",
        "case",
        "per",
        "pakistan",
        "authority",
        "office",
    ]
)
COMMON_WORDS = URDU_COMMON | ENGLISH_COMMON

# Letters (and marks) that can never begin an Urdu word.
URDU_NON_INITIAL = frozenset("ںےۓھًٌٍَُِّْٰٕٔ")

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_ARABIC_WORD = re.compile(r"[؀-ۿݐ-ݿ]+")


@dataclass(frozen=True, slots=True)
class QualityReport:
    chars: int
    words: int
    script_ratio: float
    dict_hit: float
    urdu_scramble: float
    ok: bool
    reason: str | None = None
    judged: bool = False  # enough words for the vocabulary checks to mean something


def urdu_scramble_ratio(text: str) -> tuple[float, int]:
    """Fraction of Arabic-script words that are letter-order scrambled.

    A word counts as scrambled when it starts with a letter that cannot start an Urdu word
    ("ںیم", "ےک"), or when it is a common Urdu word spelled backwards ("ربمن" for "نمبر" is not
    caught by the first rule, but "ےہ" for "ہے" is caught by both). Returns (ratio, words checked).
    """
    words = _ARABIC_WORD.findall(strip_diacritics(normalize_urdu(text)))
    words = [w for w in words if ARABIC_LETTER.search(w)]
    if not words:
        return 0.0, 0
    bad = sum(
        1 for w in words if w[0] in URDU_NON_INITIAL or (w not in URDU_COMMON and w[::-1] in URDU_COMMON)
    )
    return bad / len(words), len(words)


def assess(
    text: str,
    *,
    min_script_ratio: float,
    min_dict_hit: float,
    expected_script: str | None = None,
) -> QualityReport:
    """Judge one page of extracted text.

    `expected_script` ("Latn" or "Arab") is used when the document language is known. Without it,
    the script check only asks that letters belong to Latin or Arabic script at all (mojibake in
    other scripts fails), so bilingual pages are not penalised.
    """
    text = unicodedata.normalize("NFKC", text)
    stripped = "".join(text.split())
    letters = [c for c in stripped if c.isalpha()]
    latin = len(LATIN_LETTER.findall(stripped))
    arabic = len(ARABIC_LETTER.findall(stripped))
    if expected_script == "Arab":
        script_hits = arabic
    elif expected_script == "Latn":
        script_hits = latin
    else:
        script_hits = latin + arabic
    script_ratio = script_hits / len(letters) if letters else 0.0
    words = [w.lower() for w in _WORD.findall(strip_diacritics(normalize_urdu(text)))]
    dict_hit = sum(1 for w in words if w in COMMON_WORDS) / len(words) if words else 0.0
    scramble, arabic_words = urdu_scramble_ratio(text)

    reason: str | None = None
    if len(stripped) < MIN_PAGE_CHARS:
        reason = "too_short"
    elif script_ratio < min_script_ratio:
        reason = "wrong_script"
    elif arabic_words >= MIN_ARABIC_WORDS_FOR_SCRAMBLE_CHECK and scramble > URDU_SCRAMBLE_MAX:
        reason = "scrambled_urdu"
    elif len(words) >= MIN_WORDS_FOR_DICT_CHECK and dict_hit < min_dict_hit:
        reason = "low_dict_hit"
    return QualityReport(
        chars=len(stripped),
        words=len(words),
        script_ratio=round(script_ratio, 3),
        dict_hit=round(dict_hit, 3),
        urdu_scramble=round(scramble, 3),
        ok=reason is None,
        reason=reason,
        judged=len(words) >= MIN_WORDS_FOR_DICT_CHECK,
    )
