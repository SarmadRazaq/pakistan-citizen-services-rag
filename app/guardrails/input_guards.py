"""Input guardrails GI-1..GI-5 (blueprint §7.1). Each failure raises `AppError`.

Layers for injection / jailbreak:
1. pattern lists in English, Urdu and Roman Urdu (`patterns/injection.yaml`, `patterns/jailbreak.yaml`),
   matched on the sanitised text and on decoded variants (base64 payloads, rot13, leetspeak,
   Cyrillic/Greek homoglyphs);
2. an optional `InjectionClassifier` (hosted NemoGuard jailbreak-detect, see
   `build_injection_classifier`) above a threshold;
3. the helper model's `injection_suspect` flag, recorded as a guard flag by the pipeline.

GI-6 (off-topic), GI-7 (no documents) and GI-9 (foreign doc_ids) live in the pipeline/service,
GI-8 (rate limiting) in the app wiring.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass, field

import structlog

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.generation.prompts import message
from app.generation.query_understanding import detect_language
from app.guardrails.patterns import GuardPattern, first_match, load_patterns
from app.protocols import InjectionClassifier

log = structlog.get_logger(__name__)

OBFUSCATION_MAX_RATIO = 0.10
INJECTION_CLASSIFIER_THRESHOLD = 0.9  # Prompt Guard score above which a question is blocked (tuned in §10.3)

# Not all of these are Unicode category Cf, so they are listed explicitly.
_EXTRA_INVISIBLE = frozenset("͏ᅟᅠ឴឵᠎⠀ㅤﾠ" + "".join(chr(c) for c in range(0xFE00, 0xFE10)))
_KEEP_CONTROLS = frozenset("\n\t")

# Latin look-alikes from Cyrillic and Greek, applied only to the matching variant (never to the answer).
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
        "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "ո": "n", "А": "A", "В": "B", "Е": "E",
        "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "ο": "o",
        "α": "a", "ε": "e", "ι": "i", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "Ι": "I", "Ο": "O",
    }
)  # fmt: skip
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{20,}={0,2}(?![A-Za-z0-9+/=_-])")
_WORD = re.compile(r"[A-Za-z؀-ۿ]{2,}")


@dataclass(frozen=True, slots=True)
class InputGuardResult:
    text: str  # sanitised question: what the rest of the pipeline sees
    status: str = "passed"  # GuardrailInfo.input: passed | sanitized
    flags: tuple[str, ...] = field(default_factory=tuple)


def build_injection_classifier(settings: Settings) -> InjectionClassifier | None:
    """The hosted NemoGuard jailbreak classifier, or None (provider `none`, or no NVIDIA key)."""
    if settings.injection_provider != "nvidia" or settings.nvidia_api_key is None:
        return None
    from app.retrieval.hosted import NvidiaJailbreakClassifier

    return NvidiaJailbreakClassifier(settings)


def _is_invisible(ch: str) -> bool:
    if ch in _KEEP_CONTROLS:
        return False
    if ch in _EXTRA_INVISIBLE or 0xE0000 <= ord(ch) <= 0xE007F or 0xE0100 <= ord(ch) <= 0xE01EF:
        return True
    return unicodedata.category(ch) in {"Cf", "Cc", "Co", "Cn"}


def sanitize(text: str) -> tuple[str, int]:
    """Remove invisible / bidi / zero-width / control characters, then NFKC. Returns (text, removed)."""
    kept: list[str] = []
    removed = 0
    for ch in text:
        if ch == "\r":
            kept.append("\n")
        elif _is_invisible(ch):
            removed += 1
        else:
            kept.append(ch)
    return unicodedata.normalize("NFKC", "".join(kept)), removed


def _decoded_base64(text: str) -> list[str]:
    """Decode base64-looking tokens that turn into readable text (an encoded instruction)."""
    out: list[str] = []
    for token in _BASE64_TOKEN.findall(text):
        raw = token.replace("-", "+").replace("_", "/")
        raw += "=" * (-len(raw) % 4)
        try:
            decoded = base64.b64decode(raw, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        printable = sum(ch.isprintable() or ch.isspace() for ch in decoded)
        if decoded and printable / len(decoded) >= 0.95 and len(_WORD.findall(decoded)) >= 2:
            out.append(decoded)
    return out


def _variants(text: str) -> list[str]:
    """Texts to match: the question, its homoglyph/leet-folded forms and decoded payloads."""
    folded = text.translate(_HOMOGLYPHS)
    variants = [text, folded, folded.translate(_LEET), codecs.encode(folded, "rot13")]
    return list(dict.fromkeys(variants))


def _match_any(patterns: tuple[GuardPattern, ...], texts: list[str]) -> GuardPattern | None:
    for candidate in texts:
        hit = first_match(patterns, candidate)
        if hit is not None:
            return hit
    return None


def detect_attack(text: str) -> tuple[ErrorCode, str] | None:
    """Pattern layer: (error code, rule id) when the text is a jailbreak or injection attempt."""
    texts = _variants(text)
    decoded = _decoded_base64(text)
    for payload in decoded:
        texts.extend(_variants(payload))
    jailbreak = _match_any(load_patterns("jailbreak.yaml"), texts)
    if jailbreak is not None:
        return ErrorCode.JAILBREAK_DETECTED, jailbreak.id
    injection = _match_any(load_patterns("injection.yaml"), texts)
    if injection is not None:
        return ErrorCode.PROMPT_INJECTION_DETECTED, injection.id
    if decoded:
        # No citizen-services question carries an encoded sentence: treat it as a smuggled instruction.
        return ErrorCode.PROMPT_INJECTION_DETECTED, "encoded_payload"
    return None


async def check_question(
    question: str,
    *,
    max_chars: int,
    classifier: InjectionClassifier | None = None,
    classifier_threshold: float = INJECTION_CLASSIFIER_THRESHOLD,
) -> InputGuardResult:
    """Run GI-1..GI-5 and return the sanitised question, or raise AppError."""
    language = detect_language(question)
    stripped = question.strip()
    if not stripped:  # GI-1
        raise AppError(ErrorCode.QUESTION_EMPTY, message("question_empty", language))
    if len(stripped) > max_chars:  # GI-2
        raise AppError(
            ErrorCode.QUESTION_TOO_LONG,
            message("question_too_long", language, limit=max_chars),
            extra={"max_chars": max_chars},
        )

    text, removed = sanitize(stripped)  # GI-3
    flags: list[str] = []
    if removed:
        flags.append("invisible_chars_removed")
        if removed / max(len(stripped), 1) > OBFUSCATION_MAX_RATIO:
            log.warning("security.obfuscated_input", removed=removed, length=len(stripped))
            raise AppError(ErrorCode.OBFUSCATED_INPUT, message("obfuscated_input", language))
    text = text.strip()
    if not text:
        raise AppError(ErrorCode.QUESTION_EMPTY, message("question_empty", language))

    attack = detect_attack(text)  # GI-4 / GI-5, layer 1
    if attack is not None:
        code, rule = attack
        log.warning("security.input_blocked", code=code.slug, rule=rule)
        raise AppError(code, message(code.slug, language))

    if classifier is not None:  # layer 2 (stretch): fail open, but log, if the classifier errors
        try:
            score = await classifier.score(text)
        except Exception as exc:
            log.warning("guard.classifier_failed", error=type(exc).__name__)
        else:
            if score > classifier_threshold:
                log.warning("security.input_blocked", code="prompt_injection_detected", rule="classifier")
                raise AppError(
                    ErrorCode.PROMPT_INJECTION_DETECTED, message("prompt_injection_detected", language)
                )

    return InputGuardResult(text=text, status="sanitized" if removed else "passed", flags=tuple(flags))
