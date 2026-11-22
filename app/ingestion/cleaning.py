"""Display-text cleaning, applied to every extracted block before chunking (blueprint §3.2, §4).

The cleaned text is what users see and what citations quote, so it keeps the original digits and
diacritics. Hash/index normalisation (digits → ASCII, no diacritics) is `language.normalize_for_index`.
"""

from __future__ import annotations

import re
from dataclasses import replace

from app.ingestion.language import normalize_urdu
from app.protocols import ExtractedBlock

_HYPHEN_BREAK = re.compile(r"([A-Za-z]{2,})-\n([a-z]{2,})")
_SPACES = re.compile(r"[ \t  -   　]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_SPACE_BEFORE_PUNCT = re.compile(r"[ ]+([,.;:!?،۔؟])")


def clean_text(text: str) -> str:
    """NFKC, invisible/bidi controls removed, Urdu letters unified, whitespace tidied."""
    text = normalize_urdu(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # "regis-\ntration" → "registration" (Latin only)
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def clean_table(markdown: str) -> str:
    """Tables keep their line structure; only cell text is normalised."""
    lines = [normalize_urdu(line).strip() for line in markdown.replace("\r\n", "\n").split("\n")]
    return "\n".join(_SPACES.sub(" ", line) for line in lines if line)


def clean_blocks(blocks: list[ExtractedBlock] | tuple[ExtractedBlock, ...]) -> list[ExtractedBlock]:
    """Clean every block, drop the empty ones and recompute `char_start` in the joined text.

    The document text is `"\\n\\n".join(block.text for block in blocks)`, so after this call
    `full_text[b.char_start : b.char_start + len(b.text)] == b.text` for every block.
    """
    out: list[ExtractedBlock] = []
    pos = 0
    for block in blocks:
        text = clean_table(block.text) if block.kind == "table" else clean_text(block.text)
        if not text:
            continue
        section = tuple(s for s in (clean_text(p) for p in block.section_path) if s)
        out.append(replace(block, text=text, section_path=section, char_start=pos))
        pos += len(text) + 2
    return out


def join_blocks(blocks: list[ExtractedBlock] | tuple[ExtractedBlock, ...]) -> str:
    return "\n\n".join(b.text for b in blocks)
