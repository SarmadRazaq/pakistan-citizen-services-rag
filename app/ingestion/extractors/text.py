"""Plain-text extraction: charset detection (UTF-8/16, cp1256 for legacy Urdu), blank-line paragraphs."""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.extractors.base import decode_text
from app.protocols import ExtractedBlock, ExtractedDocument

_PARAGRAPHS = re.compile(r"\n\s*\n")


class TextExtractor:
    file_types = frozenset({"txt"})

    def extract(self, path: Path) -> ExtractedDocument:
        text = decode_text(path.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
        blocks = tuple(
            ExtractedBlock(text=para.strip(), kind="paragraph")
            for para in _PARAGRAPHS.split(text)
            if para.strip()
        )
        first = blocks[0].text.split("\n", 1)[0][:120] if blocks else None
        return ExtractedDocument(blocks=blocks, page_count=1, title=first)
