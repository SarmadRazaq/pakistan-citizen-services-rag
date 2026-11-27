"""Metadata inference for user uploads (blueprint §5.3).

Seeded documents get agency, jurisdiction and date from the corpus manifest. For uploads these
are inferred from the file name and the first pages:
- agency: a small alias dictionary ("National Database and Registration Authority", "نادرا" → NADRA);
- jurisdiction: from the agency, else phrases like "Government of the Punjab" / "حکومت سندھ";
- effective date: notification-date patterns ("Date of issue: 18 Sep 2026", "dated 12-03-2025",
  "w.e.f. …", "مورخہ …"), Urdu digits included. The PDF extractor reads the page header *before*
  stripping repeated header lines, and its date wins over one found in the body.
"""

from __future__ import annotations

import re
from datetime import date

from app.ingestion.language import normalize_digits, normalize_urdu

# canonical agency → (aliases, jurisdiction). Aliases are matched case-insensitively on word boundaries.
AGENCIES: dict[str, tuple[tuple[str, ...], str]] = {
    "NADRA": (("NADRA", "National Database and Registration Authority", "نادرا"), "federal"),
    "DGIP": (
        (
            "DGIP",
            "DGI&P",
            "Directorate General of Immigration",
            "Immigration & Passports",
            "Immigration and Passports",
            "امیگریشن اینڈ پاسپورٹس",
        ),
        "federal",
    ),
    "FBR": (("FBR", "Federal Board of Revenue", "Inland Revenue", "فیڈرل بورڈ آف ریونیو"), "federal"),
    "BISP": (("BISP", "Benazir Income Support", "بینظیر انکم سپورٹ"), "federal"),
    "Mohtasib": (
        ("Wafaqi Mohtasib", "Federal Ombudsman", "Mohtasib", "Ombudsman", "وفاقی محتسب", "محتسب"),
        "federal",
    ),
    "PMDU": (
        ("PMDU", "Prime Minister's Delivery Unit", "Pakistan Citizen Portal", "Citizen Portal"),
        "federal",
    ),
    "MoHR": (("Ministry of Human Rights", "MoHR", "وزارت انسانی حقوق"), "federal"),
    "MoLJ": (
        ("Pakistan Code", "Ministry of Law and Justice", "Ministry of Law", "National Assembly of Pakistan"),
        "federal",
    ),
    "PM Health": (
        ("Sehat Sahulat", "PM National Health", "Prime Minister National Health", "pmhealthprogram"),
        "federal",
    ),
    "KP Sehat": (("Sehat Card Plus", "sehatcardplus"), "kp"),
    "Punjab": (
        (
            "Government of the Punjab",
            "Government of Punjab",
            "Punjab Consumer Protection",
            "DLIMS",
            "City Traffic Police Lahore",
            "punjabcode",
            "حکومت پنجاب",
        ),
        "punjab",
    ),
    "Sindh": (
        (
            "Government of Sindh",
            "Sindh Consumer Protection",
            "Sindh Police",
            "Sindh Judicial Academy",
            "sindhlaws",
            "حکومت سندھ",
        ),
        "sindh",
    ),
}

_JURISDICTION_PHRASES: tuple[tuple[str, str], ...] = (
    ("punjab", r"\bpunjab\b|پنجاب"),
    ("sindh", r"\bsindh\b|سندھ"),
    ("kp", r"khyber\s+pakhtunkhwa|\bk\.?p\.?k?\b|خیبر\s*پختونخوا"),
    ("federal", r"government\s+of\s+pakistan|islamabad|federal|حکومت\s*پاکستان|وفاقی"),
)

_TOPICS: tuple[tuple[str, str], ...] = (
    ("passport", r"passport|پاسپورٹ"),
    ("identity", r"\bcnic\b|\bnicop\b|identity\s+card|\bcrc\b|juvenile|family\s+registration|شناختی"),
    ("tax", r"income\s+tax|withholding|\bfbr\b|tax\s*payer|ٹیکس"),
    ("driving_licence", r"driving\s+licen[cs]e|\bdlims\b|motor\s+vehicles|ڈرائیونگ"),
    ("consumer_protection", r"consumer|صارف"),
    ("health", r"sehat|health|صحت"),
    ("complaints", r"mohtasib|ombudsman|complaint|citizen\s+portal|شکایت"),
    ("rights", r"constitution|fundamental\s+rights|human\s+rights|آئین"),
    ("social_protection", r"\bbisp\b|kafaalat|benazir"),
)
TOPICS: tuple[str, ...] = tuple(t for t, _ in _TOPICS)  # the closed Topic list (GLOSSARY.md "Topic")


def normalize_topic(value: str) -> str:
    """Slug form of a topic: trimmed, case-folded, runs of whitespace/hyphens joined by one `_`."""
    return re.sub(r"[\s\-]+", "_", value.strip().casefold())


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias)
    if alias.isascii():
        return re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", re.IGNORECASE)
    return re.compile(escaped)


_ALIAS_PATTERNS = {agency: [_alias_pattern(a) for a in aliases] for agency, (aliases, _) in AGENCIES.items()}


def infer_agency(file_name: str, text: str) -> str | None:
    """Best-matching agency; matches in the file name weigh three times as much as in the text."""
    text = normalize_urdu(text[:20_000])
    name = re.sub(r"[_\-]+", " ", file_name)
    scores: dict[str, float] = {}
    for agency, patterns in _ALIAS_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            score += 3.0 * len(pattern.findall(name)) + len(pattern.findall(text))
        if score:
            scores[agency] = score
    if not scores:
        return None
    return max(scores, key=lambda a: scores[a])


def jurisdiction_for_agency(agency: str | None) -> str | None:
    if agency is None:
        return None
    entry = AGENCIES.get(agency)
    return entry[1] if entry else None


def infer_jurisdiction(agency: str | None, text: str) -> str | None:
    from_agency = jurisdiction_for_agency(agency)
    if from_agency:
        return from_agency
    text = normalize_urdu(text[:20_000])
    counts = {j: len(re.findall(p, text, re.IGNORECASE)) for j, p in _JURISDICTION_PHRASES}
    best = max(counts, key=lambda j: counts[j])
    return best if counts[best] else None


def infer_topic(file_name: str, text: str) -> str | None:
    haystack = f"{file_name} {normalize_urdu(text[:20_000])}"
    counts = {t: len(re.findall(p, haystack, re.IGNORECASE)) for t, p in _TOPICS}
    best = max(counts, key=lambda t: counts[t])
    return best if counts[best] else None


# ---------------------------------------------------------------- dates

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_URDU_MONTHS = {
    "جنوری": 1, "فروری": 2, "مارچ": 3, "اپریل": 4, "مئی": 5, "جون": 6, "جولائی": 7, "اگست": 8,
    "ستمبر": 9, "اکتوبر": 10, "نومبر": 11, "دسمبر": 12,
}  # fmt: skip
_MONTH_NAME = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_URDU_MONTH_NAME = "(?:" + "|".join(_URDU_MONTHS) + ")"
_DATE = (
    r"(?P<date>"
    r"\d{4}-\d{1,2}-\d{1,2}"  # ISO
    r"|\d{1,2}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{2,4}"  # 12-03-2025 (day first, as in Pakistan)
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s*(?:of\s+)?{_MONTH_NAME}\.?,?\s*\d{{4}}"  # 18 Sep 2026
    rf"|{_MONTH_NAME}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}"  # September 18, 2026
    rf"|\d{{1,2}}\s*{_URDU_MONTH_NAME}\s*،?\s*\d{{4}}"  # ۱۸ ستمبر ۲۰۲۶ (digits normalised)
    r")"
)
_CUES = (
    r"date\s+of\s+issue",
    r"issue\s+date",
    r"dated",
    r"w\.?\s*e\.?\s*f\.?",
    r"with\s+effect\s+from",
    r"effective\s+(?:from|date|on)",
    r"as\s+on",
    r"amended\s+up\s*to",
    r"updated\s+(?:on|up\s*to)",
    r"notified\s+on",
    r"مورخہ",
    r"تاریخ\s*اجرا[ءئ]?",
    r"بتاریخ",
)
_CUE_DATE = re.compile(rf"(?:{'|'.join(_CUES)})\s*[:\-–—]?\s*{_DATE}", re.IGNORECASE)
_ANY_DATE = re.compile(_DATE, re.IGNORECASE)


def parse_date(raw: str) -> date | None:
    """Parse one date string in the formats above; day-first for numeric dates. None if invalid."""
    s = normalize_digits(normalize_urdu(raw)).strip().lower().rstrip(".,")
    try:
        if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
            y, mo, d = int(m[1]), int(m[2]), int(m[3])
        elif m := re.fullmatch(r"(\d{1,2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{2,4})", s):
            d, mo, y = int(m[1]), int(m[2]), int(m[3])
            if y < 100:
                y += 2000 if y < 70 else 1900
            if mo > 12 and d <= 12:  # clearly month-first (e.g. 03/25/2025)
                d, mo = mo, d
        elif m := re.fullmatch(
            rf"(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_NAME})\.?,?\s*(\d{{4}})", s
        ):
            d, mo, y = int(m[1]), _MONTHS[m[2][:3]], int(m[3])
        elif m := re.fullmatch(rf"({_MONTH_NAME})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})", s):
            mo, d, y = _MONTHS[m[1][:3]], int(m[2]), int(m[3])
        elif m := re.fullmatch(rf"(\d{{1,2}})\s*({_URDU_MONTH_NAME})\s*،?\s*(\d{{4}})", s):
            d, mo, y = int(m[1]), _URDU_MONTHS[m[2]], int(m[3])
        else:
            return None
        if not 1947 <= y <= 2100:
            return None
        return date(y, mo, d)
    except (ValueError, KeyError):
        return None


def find_effective_date(text: str) -> date | None:
    """The first cue-word date ("Date of issue: …", "dated …", "مورخہ …") in the text, if any."""
    text = normalize_digits(normalize_urdu(text))
    for m in _CUE_DATE.finditer(text):
        if parsed := parse_date(m.group("date")):
            return parsed
    return None


def find_any_date(text: str) -> date | None:
    """The first parseable date anywhere (used for manifest cells such as "Gazette 14 Feb 2022")."""
    text = normalize_digits(normalize_urdu(text))
    for m in _ANY_DATE.finditer(text):
        if parsed := parse_date(m.group("date")):
            return parsed
    return None
