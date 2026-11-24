"""GO-9 "information, not advice" (blueprint §7.2, E44): advice-request patterns and verified referrals.

The code, never the model, adds the not-advice notice and the referral contact. Contacts come only
from `referrals.yaml` entries marked `verified: true`; unverified entries are dropped at load time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped,unused-ignore]

from app.guardrails.patterns import first_match, load_patterns
from app.ingestion.metadata import TOPICS

log = structlog.get_logger(__name__)

ADVICE_FILE = "advice.yaml"
# Referral triggers (GLOSSARY.md): found in the question itself, never in a Document's Topic.
TRIGGERS = frozenset({"emergency", "legal_advice"})


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
    triggers: frozenset[str] = frozenset()
    source_url: str | None = None


def _names(entry: dict[str, Any], key: str, allowed: Iterable[str]) -> frozenset[str]:
    """The entry's `topics` or `triggers`, keeping only known names (unknown ones are logged)."""
    names = {str(t).strip().lower() for t in entry.get(key) or []}
    known = names & set(allowed)
    if names - known:
        log.warning("referrals.unknown_names", entry=entry.get("id"), field=key, names=sorted(names - known))
    return frozenset(known)


def parse_referrals(data: object) -> tuple[Referral, ...]:
    """Keep only entries with `verified: true` and a non-empty contact (a Referral, GLOSSARY.md)."""
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
                topics=_names(entry, "topics", TOPICS),
                triggers=_names(entry, "triggers", TRIGGERS),
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
    """Topic + jurisdiction match first; otherwise a legal_advice referral; otherwise nothing.

    Emergency referrals are never matched here: they belong to the emergency Notice, not to advice.
    """
    wanted_topics = {t.lower() for t in topics if t}
    wanted_places = {j.lower() for j in jurisdictions if j} | {"federal"}
    pool = [r for r in referrals if r.jurisdiction in wanted_places and "emergency" not in r.triggers]
    exact = [r for r in pool if r.topics & wanted_topics]
    if exact:
        return exact[:limit]
    generic = [r for r in pool if "legal_advice" in r.triggers]
    return generic[:limit]


def match_emergency_referrals(
    referrals: Iterable[Referral], *, jurisdictions: Iterable[str]
) -> list[Referral]:
    """Verified Referrals with the `emergency` trigger in the given jurisdictions (federal always)."""
    wanted_places = {j.lower() for j in jurisdictions if j} | {"federal"}
    return [r for r in referrals if "emergency" in r.triggers and r.jurisdiction in wanted_places]
