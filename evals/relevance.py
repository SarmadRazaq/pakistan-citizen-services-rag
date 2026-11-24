"""Is a retrieved chunk relevant to a golden item? (blueprint §10.1, "key choice")

Gold labels name a Source (its seed ID) + page/section, so they stay valid across chunking configs,
re-collection and renamed files.

A chunk is **relevant** (grade 2) to a gold source when
1. its file comes from the gold Source: corpus files are named `<ID>__slug.ext`, and the ID prefix
   must equal the gold `source_id` exactly (case-insensitive; `ID-02a` and `ID-02b` are different
   Sources, and so are the Translation pair `ID-05` and `ID-06`), **and**
2. its page span overlaps the gold pages, **or** (HTML/DOCX/MD) its section path matches the gold section.

A chunk from the right Source at the wrong place gets grade 1 (used by nDCG only). A gold source with
neither pages nor a section is a whole-Source label: any chunk of that Source is grade 2.

Chunks are plain dicts (the JSON the API returns), so this module does not depend on the service code.
Page spans are read from `page_start`/`page_end` when present, else `page`; sections from
`section_path` (list) or `section` ("A › B"). When a chunk carries no section at all, the gold heading
appearing in the chunk text counts as a section match (contextual chunk headers put it there).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from evals.golden import GoldSource

_ID_PREFIX = re.compile(r"^(?P<id>[a-z]{2,4}-\d+[a-z]?)__(?P<slug>.+)$")
_SECTION_SEP = re.compile(r"\s*(?:›|»|>|/|\|)\s*")
_SPACE = re.compile(r"\s+")

Chunk = Mapping[str, Any]


def _basename(name: str) -> str:
    return re.split(r"[\\/]", name.strip())[-1].casefold()


def split_source_id(file_name: str) -> tuple[str | None, str]:
    """`ID-05__slug.pdf` -> ("id-05", "slug.pdf"); a plain name -> (None, name). Case-insensitive."""
    base = _basename(file_name)
    m = _ID_PREFIX.match(base)
    if m:
        return m.group("id"), m.group("slug")
    return None, base


def source_matches(chunk_file: str | None, source_id: str) -> bool:
    """True when the chunk's file was collected from the Source `source_id`."""
    if not chunk_file:
        return False
    chunk_id, _ = split_source_id(chunk_file)
    return chunk_id is not None and chunk_id == source_id.strip().casefold()


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def chunk_pages(chunk: Chunk) -> set[int]:
    """Pages a chunk spans (1-based, as the API reports them). Empty when unknown."""
    pages = chunk.get("pages")
    if isinstance(pages, list):
        return {p for p in (_as_int(x) for x in pages) if p is not None}
    start = _as_int(chunk.get("page_start"))
    end = _as_int(chunk.get("page_end"))
    if start is None:
        start = _as_int(chunk.get("page"))
    if start is None:
        return set()
    if end is None or end < start:
        end = start
    return set(range(start, end + 1))


def _norm_section(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def chunk_sections(chunk: Chunk) -> list[str]:
    """The chunk's section path as normalized parts (outermost first)."""
    path = chunk.get("section_path")
    if isinstance(path, list) and path:
        return [_norm_section(str(p)) for p in path if str(p).strip()]
    section = chunk.get("section")
    if isinstance(section, str) and section.strip():
        return [_norm_section(p) for p in _SECTION_SEP.split(section) if p.strip()]
    return []


def section_matches(chunk: Chunk, gold_section: str) -> bool:
    gold_parts = [_norm_section(p) for p in _SECTION_SEP.split(gold_section) if p.strip()]
    if not gold_parts:
        return False
    parts = chunk_sections(chunk)
    if parts:
        if gold_parts == parts[: len(gold_parts)]:  # gold path is a prefix of the chunk's path
            return True
        if len(gold_parts) == 1 and gold_parts[0] in parts:  # a single heading anywhere in the path
            return True
        return " › ".join(gold_parts) in " › ".join(parts)
    text = chunk.get("text")
    return isinstance(text, str) and gold_parts[-1] in _norm_section(text)


def grade(chunk: Chunk, gold: GoldSource) -> int:
    """2 = right Source and right place, 1 = right Source only, 0 = other Source."""
    if not source_matches(chunk.get("file_name"), gold.source_id):
        return 0
    if not gold.pages and not gold.section:
        return 2
    if gold.pages and chunk_pages(chunk) & set(gold.pages):
        return 2
    if gold.section and section_matches(chunk, gold.section):
        return 2
    return 1


def chunk_grade(chunk: Chunk, golds: Sequence[GoldSource]) -> int:
    return max((grade(chunk, g) for g in golds), default=0)


def matched_gold(chunk: Chunk, golds: Sequence[GoldSource]) -> set[int]:
    """Indices of the gold sources this chunk is fully relevant to (used by Recall@k)."""
    return {i for i, g in enumerate(golds) if grade(chunk, g) == 2}


def is_relevant(chunk: Chunk, golds: Sequence[GoldSource]) -> bool:
    return chunk_grade(chunk, golds) == 2
