"""Markdown extraction: ATX headings → `section_path`, pipe tables kept as table blocks, fenced code kept."""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.extractors.base import SectionTracker, decode_text
from app.protocols import ExtractedBlock, ExtractedDocument

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)")
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


class MarkdownExtractor:
    file_types = frozenset({"md"})

    def extract(self, path: Path) -> ExtractedDocument:
        text = decode_text(path.read_bytes(), "Markdown file").replace("\r\n", "\n").replace("\r", "\n")
        tracker = SectionTracker()
        blocks: list[ExtractedBlock] = []
        buffer: list[str] = []
        kind = "paragraph"
        title: str | None = None
        in_fence = False

        def flush() -> None:
            nonlocal buffer, kind
            body = "\n".join(buffer).strip()
            if body:
                blocks.append(ExtractedBlock(text=body, kind=kind, section_path=tracker.path))  # type: ignore[arg-type]
            buffer = []
            kind = "paragraph"

        for line in text.split("\n"):
            if _FENCE.match(line.strip()):
                if not in_fence:
                    flush()
                buffer.append(line)
                in_fence = not in_fence
                if not in_fence:
                    flush()
                continue
            if in_fence:
                buffer.append(line)
                continue
            if m := _ATX.match(line):
                flush()
                level, heading = len(m.group(1)), m.group(2).strip()
                if heading:
                    path_ = tracker.enter(level, heading)
                    title = title or (heading if level == 1 else None)
                    blocks.append(ExtractedBlock(text=heading, kind="heading", section_path=path_))
                continue
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            line_kind = "table" if stripped.startswith("|") else "list" if _LIST.match(line) else "paragraph"
            if buffer and line_kind != kind and not (kind == "list" and line_kind == "paragraph"):
                flush()
            if not buffer:
                kind = line_kind
            buffer.append(stripped if line_kind == "table" else line.rstrip())
        flush()
        return ExtractedDocument(blocks=tuple(blocks), page_count=1, title=title)
