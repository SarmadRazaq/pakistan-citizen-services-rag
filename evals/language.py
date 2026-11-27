"""A small, dependency-free answer-language detector: en / ur (Arabic script) / roman_ur.

The eval checks the *answer* language independently of what the service claims in `language.answer`,
so it cannot import the service's detector. The rules are deliberately simple and transparent:

- Urdu: at least 50% of the letters are Arabic-script.
- Roman Urdu vs English: count Urdu function words written in Latin script against English function
  words. Code-switched Roman Urdu ("CNIC ke liye marriage certificate chahiye") keeps its Urdu function
  words, so it is still detected as Roman Urdu. Ties go to English.
"""

from __future__ import annotations

import re
from typing import Literal

DetectedLanguage = Literal["en", "ur", "roman_ur", "unknown"]

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-z']+")
_MARKER = re.compile(r"\[S\d+\]")


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Latin-script Urdu function words and very common verbs. Words that are also common English words
# ("to", "main", "par", "me", "so", "the") are left out on purpose.
ROMAN_URDU_WORDS = _words(
    """
    hai hain ho hoga hogi honge hota hoti hote tha thi thay ka ki ke ko se mein mai mujhe aap ap
    apna apni apne aur ya nahi nahin kya kaise kaisay kab kahan kyun kyon kitna kitni kitne karein
    karen karna karni karne kar karwana karwayen karwaen karwaun sakte sakta sakti saktay liye liay
    lie wala wali walay wale agar toh bhi yeh ye woh wo iss uss jab tak sirf lekin magar chahiye
    chahye zaroori zaruri dein dain lein jata jati jate jaata jaati jaate jayega jaegi hua hui hue
    gaya gayi gaye diya diye liya kiya kiye milta milti milte banwane banwana banta baad pehle
    pehlay saath sath bare baray mutabiq zariye zariya taraf dobara waqt saal
    """
)

ENGLISH_WORDS = _words(
    """
    the is are was were be been of and or to in on for with by from at as an a this that these those
    it its you your can cannot must should will would not no yes if then than there their they which
    what when where who how do does did have has had any all each only also may might per under
    """
)


def detect_language(text: str) -> DetectedLanguage:
    clean = _MARKER.sub(" ", text or "")
    arabic = len(_ARABIC.findall(clean))
    latin = len(_LATIN.findall(clean))
    if arabic + latin == 0:
        return "unknown"
    if arabic / (arabic + latin) >= 0.5:
        return "ur"
    words = [w.casefold() for w in _WORD.findall(clean)]
    roman = sum(w in ROMAN_URDU_WORDS for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    return "roman_ur" if roman > english else "en"
