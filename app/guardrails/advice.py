"""GO-9 "information, not advice" (blueprint §7.2, E44): advice-request patterns and verified referrals.

The code, never the model, adds the not-advice notice and the referral contact. Contacts come only
from `referrals.yaml` entries marked `verified: true`; unverified entries are dropped at load time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import structlog
import yaml  # type: ignore[import-untyped,unused-ignore]

from app.guardrails.patterns import first_match, load_patterns

log = structlog.get_logger(__name__)

ADVICE_FILE = "advice.yaml"
GENERIC_ADVICE_TOPICS = frozenset({"advice_referral", "legal_advice"})


def is_advice_request(question: str) -> bool:
    return first_match(load_patterns(ADVICE_FILE, "request"), question) is not None


def has_directive_advice(answer: str) -> bool:
    """True when the answer tells the user what they personally should do ("you should file…")."""
    return first_match(load_patterns(ADVICE_FILE, "directive"), answer) is not None


@dataclass(frozen=True, slots=True)
class Referral:
    id: str
    name: str
    contact: str
    jurisdiction: str
    topics: frozenset[str]
    source_url: str | None = None


def parse_referrals(data: object) -> tuple[Referral, ...]:
    """Keep only entries with `verified: true` and a non-empty contact."""
    if not isinstance(data, dict):
        return ()
    out: list[Referral] = []
    for entry in data.get("entries") or []:
        if not isinstance(entry, dict) or entry.get("verified") is not True:
            continue
        contact = str(entry.get("contact") or "").strip()
        if not contact:
            continue
        out.append(
            Referral(
                id=str(entry.get("id", "")),
                name=str(entry.get("name", "")).strip(),
                contact=contact,
                jurisdiction=str(entry.get("jurisdiction") or "federal").lower(),
                topics=frozenset(str(t).lower() for t in entry.get("topics") or []),
                source_url=(str(entry["source_url"]).strip() or None) if entry.get("source_url") else None,
            )
        )
    return tuple(out)


@cache
def load_referrals(path: Path) -> tuple[Referral, ...]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        log.warning("referrals.load_failed", path=str(path), error=type(exc).__name__)
        return ()
    referrals = parse_referrals(data)
    log.info("referrals.loaded", verified=len(referrals))
    return referrals


def match_referrals(
    referrals: Iterable[Referral],
    *,
    topics: Iterable[str],
    jurisdictions: Iterable[str],
    limit: int = 2,
) -> list[Referral]:
    """Topic + jurisdiction match first; otherwise a generic legal-advice referral; otherwise nothing."""
    wanted_topics = {t.lower() for t in topics if t}
    wanted_places = {j.lower() for j in jurisdictions if j} | {"federal"}
    pool = [r for r in referrals if r.jurisdiction in wanted_places]
    exact = [r for r in pool if r.topics & wanted_topics and "emergency" not in r.topics]
    if exact:
        return exact[:limit]
    generic = [r for r in pool if r.topics & GENERIC_ADVICE_TOPICS]
    return generic[:limit]
