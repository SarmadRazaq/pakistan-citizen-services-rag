"""DOCX extraction with python-docx (blueprint §3.2).

Body paragraphs and tables in document order; "Heading N" / "Title" styles → `section_path`;
tables → Markdown. Section headers and footers live in separate parts and are skipped.
Hidden runs (`w:vanish`, white text, < 2 pt) are removed and counted (GD-9).
"""

from __future__ import annotations

import re
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.shared import RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.errors import AppError, ErrorCode
from app.ingestion.extractors.base import SectionTracker, rows_to_markdown
from app.protocols import ExtractedBlock, ExtractedDocument

_HEADING = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)
_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HIDDEN_MAX_FONT_PT = 2.0


def _visible_text(paragraph: Paragraph) -> tuple[str, int]:
    parts: list[str] = []
    hidden = 0
    for run in paragraph.runs:
        font = run.font
        is_hidden = bool(font.hidden)
        try:
            if font.color is not None and font.color.type is not None and font.color.rgb == _WHITE:
                is_hidden = True
        except (AttributeError, ValueError):
            pass
        if font.size is not None and font.size.pt < HIDDEN_MAX_FONT_PT:
            is_hidden = True
        if is_hidden:
            if run.text.strip():
                hidden += 1
            continue
        parts.append(run.text)
    text = "".join(parts) if paragraph.runs else paragraph.text
    return " ".join(text.split()), hidden


def _heading_level(paragraph: Paragraph) -> int | None:
    style = paragraph.style
    name = (style.name or "") if style is not None else ""
    if name.lower() == "title":
        return 1
    if m := _HEADING.match(name):
        return max(1, int(m.group(1)))
    return None


def _table_markdown(table: Table) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        previous = None
        for cell in row.cells:
            # python-docx repeats a merged cell once per grid column; keep it once.
            if previous is not None and cell._tc is previous:
                continue
            previous = cell._tc
            cells.append(" ".join(cell.text.split()))
        rows.append(cells)
    return rows_to_markdown(rows) if len(rows) >= 1 else ""


class DocxExtractor:
    file_types = frozenset({"docx"})

    def extract(self, path: Path) -> ExtractedDocument:
        try:
            document: DocxDocument = docx.Document(str(path))
        except Exception as exc:  # zipfile/lxml/KeyError for broken packages
            raise AppError(ErrorCode.CORRUPT_FILE, "The DOCX file is corrupt.") from exc
        tracker = SectionTracker()
        blocks: list[ExtractedBlock] = []
        hidden = 0
        title: str | None = None
        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text, removed = _visible_text(item)
                hidden += removed
                if not text:
                    continue
                level = _heading_level(item)
                if level is not None:
                    path_ = tracker.enter(level, text)
                    title = title or (text if level == 1 else None)
                    blocks.append(ExtractedBlock(text=text, kind="heading", section_path=path_))
                else:
                    style = item.style
                    is_list = style is not None and "list" in (style.name or "").lower()
                    blocks.append(
                        ExtractedBlock(
                            text=f"- {text}" if is_list else text,
                            kind="list" if is_list else "paragraph",
                            section_path=tracker.path,
                        )
                    )
            elif isinstance(item, Table):
                markdown = _table_markdown(item)
                if markdown:
                    blocks.append(ExtractedBlock(text=markdown, kind="table", section_path=tracker.path))
        core_title = (document.core_properties.title or "").strip()
        return ExtractedDocument(
            blocks=tuple(blocks),
            page_count=1,
            title=core_title or title,
            hidden_text_removed=hidden,
        )
