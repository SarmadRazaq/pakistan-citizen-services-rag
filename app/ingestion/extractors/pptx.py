"""PPTX extraction with python-pptx: slide number as page, slide title as section, tables → Markdown."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pptx import Presentation

from app.core.errors import AppError, ErrorCode
from app.ingestion.extractors.base import rows_to_markdown
from app.protocols import ExtractedBlock, ExtractedDocument


def _shapes(shapes: Any) -> list[Any]:
    out: list[Any] = []
    for shape in shapes:
        if getattr(shape, "shape_type", None) == 6 and hasattr(shape, "shapes"):  # group shape
            out.extend(_shapes(shape.shapes))
        else:
            out.append(shape)
    return out


class PptxExtractor:
    file_types = frozenset({"pptx"})

    def extract(self, path: Path) -> ExtractedDocument:
        try:
            deck = Presentation(str(path))
        except Exception as exc:
            raise AppError(ErrorCode.CORRUPT_FILE, "The PPTX file is corrupt.") from exc
        blocks: list[ExtractedBlock] = []
        first_title: str | None = None
        slide_count = 0
        for number, slide in enumerate(deck.slides, start=1):
            slide_count = number
            title_shape = slide.shapes.title
            title = " ".join(title_shape.text_frame.text.split()) if title_shape is not None else ""
            section = (title,) if title else (f"Slide {number}",)
            first_title = first_title or title or None
            if title:
                blocks.append(ExtractedBlock(text=title, kind="heading", page=number, section_path=section))
            for shape in _shapes(slide.shapes):
                if title_shape is not None and shape.shape_id == title_shape.shape_id:
                    continue
                if getattr(shape, "has_table", False) and shape.has_table:
                    rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                    markdown = rows_to_markdown(rows)
                    if markdown:
                        blocks.append(
                            ExtractedBlock(text=markdown, kind="table", page=number, section_path=section)
                        )
                elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = " ".join("".join(r.text for r in para.runs).split())
                        if text:
                            kind: Literal["list", "paragraph"] = "list" if para.level > 0 else "paragraph"
                            blocks.append(
                                ExtractedBlock(text=text, kind=kind, page=number, section_path=section)
                            )
            if slide.has_notes_slide:
                notes = " ".join(slide.notes_slide.notes_text_frame.text.split())
                if notes:
                    blocks.append(
                        ExtractedBlock(text=notes, kind="paragraph", page=number, section_path=section)
                    )
        return ExtractedDocument(blocks=tuple(blocks), page_count=slide_count, title=first_title)
