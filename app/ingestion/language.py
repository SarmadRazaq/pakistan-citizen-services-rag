"""Language, script and Urdu normalisation helpers (blueprint §4).

- Chunks get `language` ∈ {en, ur} and `script` ∈ {Latn, Arab} from their script ratio.
- Queries can also be Roman Urdu (Latin script); a small lexicon heuristic spots it.
- `normalize_urdu` unifies Arabic/Urdu letter variants; `normalize_for_index` additionally maps
  Urdu digits to ASCII and strips diacritics. That form is for hashing and indexing only: the
  stored display text keeps the original digits.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

Script = Literal["Latn", "Arab"]
ChunkLanguage = Literal["en", "ur"]
QueryLanguage = Literal["en", "ur", "roman_ur"]

# Arabic, Arabic Supplement, Arabic Extended-A and the presentation-form blocks.
ARABIC_CHAR = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
ARABIC_LETTER = re.compile(r"[ء-يٮٯٱ-ۓەۮۯۺ-ۿݐ-ݿ]")
LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")

# Zero-width and bidi control characters: invisible, and a known prompt-injection vector.
_INVISIBLE = re.compile(r"[­؜᠎​-‏‪-‮⁠-⁤⁦-⁩﻿]")
_TATWEEL = "ـ"
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")

# Arabic-script letter variants → the Urdu letter (Arabic Yeh/Kaf/Heh and friends).
_URDU_LETTERS = str.maketrans(
    {
        "ي": "ی",  # ي Arabic Yeh → ی Farsi/Urdu Yeh
        "ى": "ی",  # ى Alef Maksura → ی
        "ك": "ک",  # ك Arabic Kaf → ک Keheh
        "ه": "ہ",  # ه Arabic Heh → ہ Heh Goal
        "ە": "ہ",  # ە Ae → ہ
        "ة": "ۃ",  # ة Teh Marbuta → ۃ
    }
)
_DIGITS = str.maketrans(
    {**{chr(0x06F0 + i): str(i) for i in range(10)}, **{chr(0x0660 + i): str(i) for i in range(10)}}
)

# Roman Urdu cue words. Strong cues are not English words; weak ones also occur in English
# ("main office", "par value"), so they count half.
_ROMAN_STRONG = frozenset(
    [
        "kya",
        "kia",
        "hai",
        "hain",
        "hay",
        "hy",
        "kitni",
        "kitna",
        "kitne",
        "kitnay",
        "kaise",
        "kaisay",
        "kese",
        "kesy",
        "kahan",
        "kahaan",
        "kab",
        "kyun",
        "kyon",
        "kiun",
        "nahi",
        "nahin",
        "nai",
        "chahiye",
        "chahye",
        "chaiye",
        "banwana",
        "banwani",
        "banwane",
        "banwayein",
        "banwain",
        "karna",
        "karni",
        "karen",
        "karein",
        "karo",
        "kese",
        "hota",
        "hoti",
        "hote",
        "hoga",
        "hogi",
        "honge",
        "mera",
        "meri",
        "mere",
        "hamara",
        "humara",
        "apna",
        "apni",
        "apne",
        "liye",
        "lie",
        "wala",
        "wali",
        "wale",
        "sakta",
        "sakti",
        "sakte",
        "raha",
        "rahi",
        "rahe",
        "tha",
        "thi",
        "thay",
        "bhi",
        "sirf",
        "kaun",
        "kon",
        "konsa",
        "konsi",
        "kahin",
        "agar",
        "lekin",
        "magar",
        "abhi",
        "baad",
        "pehle",
        "pehlay",
        "dobara",
        "naya",
        "nayi",
        "purana",
        "shanakhti",
        "shnakhti",
        "karwana",
        "karwani",
        "milega",
        "milegi",
        "milta",
        "milti",
        "jata",
        "jati",
        "jate",
        "dena",
        "deni",
        "lena",
        "leni",
    ]
)
_ROMAN_WEAK = frozenset(["ki", "ka", "ke", "ko", "se", "mein", "main", "mai", "par", "aur", "ya", "to"])


def strip_invisible(text: str) -> str:
    return _INVISIBLE.sub("", text)


def normalize_urdu(text: str) -> str:
    """NFKC, invisible-character removal, tatweel removal and Urdu letter unification."""
    text = unicodedata.normalize("NFKC", text)  # also folds Arabic presentation forms to base letters
    text = strip_invisible(text).replace(_TATWEEL, "")
    return text.translate(_URDU_LETTERS)


def normalize_digits(text: str) -> str:
    """Urdu (۰–۹) and Arabic-Indic (٠–٩) digits → ASCII, so amounts and dates match across scripts."""
    return text.translate(_DIGITS)


def strip_diacritics(text: str) -> str:
    return _DIACRITICS.sub("", text)


def normalize_for_index(text: str) -> str:
    """The form used for content hashes and the sparse index (never for display)."""
    text = strip_diacritics(normalize_digits(normalize_urdu(text)))
    return " ".join(text.split())


def script_counts(text: str) -> tuple[int, int]:
    """(Latin letters, Arabic-script letters)."""
    return len(LATIN_LETTER.findall(text)), len(ARABIC_LETTER.findall(text))


def arabic_ratio(text: str) -> float:
    latin, arabic = script_counts(text)
    total = latin + arabic
    return arabic / total if total else 0.0


def detect_script(text: str) -> Script:
    return "Arab" if arabic_ratio(text) >= 0.5 else "Latn"


def detect_language(text: str) -> ChunkLanguage:
    """Chunk language from the script ratio: Arabic script → Urdu, otherwise English."""
    return "ur" if detect_script(text) == "Arab" else "en"


_LATIN_WORD = re.compile(r"[A-Za-z]+")


def roman_urdu_score(text: str) -> float:
    words = [w.lower() for w in _LATIN_WORD.findall(text)]
    return sum(1.0 if w in _ROMAN_STRONG else 0.5 if w in _ROMAN_WEAK else 0.0 for w in words)


def is_roman_urdu(text: str) -> bool:
    words = _LATIN_WORD.findall(text)
    if not words:
        return False
    score = roman_urdu_score(text)
    return score >= 1.5 or (score >= 1.0 and score / len(words) >= 0.25)


def detect_query_language(text: str) -> QueryLanguage:
    """Arabic script → ur; Latin script with Roman Urdu cue words → roman_ur; else en."""
    if detect_script(text) == "Arab":
        return "ur"
    return "roman_ur" if is_roman_urdu(text) else "en"


def majority_language(languages: list[str]) -> str | None:
    if not languages:
        return None
    return max(sorted(set(languages)), key=languages.count)
