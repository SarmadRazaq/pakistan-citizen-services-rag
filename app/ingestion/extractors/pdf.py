"""PDF extraction with PyMuPDF (blueprint §3.2).

Per page:
1. spans → lines, dropping **hidden text** (white, < 2 pt, fully transparent or off-page; GD-9);
2. `page.find_tables()` → Markdown table blocks; lines inside a table area are not repeated as prose;
3. the **effective date** is read from the page header ("Date of issue: 18 Sep 2026", "dated …")
   *before* repeated header lines are removed (§5.3), and exposed as `metadata["effective_date"]`;
4. **headers/footers** are stripped: lines in the top/bottom 8% of the page that repeat (digits
   ignored) on ≥ 50% of pages, plus bare page numbers there;
5. the **text-quality check** (E46) runs on what is left. A failing page is OCR'd from its rendered
   image when Tesseract is available; otherwise it is marked unusable and skipped.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import pymupdf
import structlog

from app.core.errors import AppError, ErrorCode
from app.ingestion.extractors.base import SectionTracker, rows_to_markdown
from app.ingestion.metadata import find_effective_date
from app.ingestion.ocr import NoOcr, PageOcr
from app.ingestion.text_quality import JUNK_REASONS, QualityReport, assess
from app.protocols import ExtractedBlock, ExtractedDocument

log = structlog.get_logger(__name__)
fitz: Any = pymupdf  # PyMuPDF's API is untyped; treat it as Any under mypy --strict

MARGIN_FRACTION = 0.08  # header/footer zone: top and bottom 8% of the page height
REPEAT_PAGE_FRACTION = 0.5  # a margin line repeated on ≥ 50% of pages is a header/footer
HEADER_DATE_FRACTION = 0.25  # the date is looked for in the top quarter of the first pages first
HIDDEN_MAX_FONT_PT = 2.0
JUNK_DOCUMENT_FRACTION = 0.5  # most judged pages junk → short pages are not trusted either
DARK_FILL_LUMINANCE = 0.85
MIN_GRID_LINES = 3

_PAGE_NUMBER = re.compile(
    r"^\W*(?:page|pg\.?|صفحہ)?\s*\d{1,4}(?:\s*(?:of|/|از)\s*\d{1,4})?\W*$|^\W*[ivxlcdm]{1,6}\W*$",
    re.IGNORECASE,
)
_DIGITS = re.compile(r"\d+")


@dataclass(slots=True)
class _Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    block_no: int


@dataclass(slots=True)
class _Table:
    y0: float
    markdown: str


@dataclass(slots=True)
class _Page:
    number: int  # 1-based
    height: float
    lines: list[_Line] = field(default_factory=list)
    tables: list[_Table] = field(default_factory=list)
    hidden: int = 0

    def in_margin(self, line: _Line, fraction: float = MARGIN_FRACTION) -> bool:
        # A header line *starts* in the top band; a footer line *ends* in the bottom band.
        return line.y0 <= self.height * fraction or line.y1 >= self.height * (1 - fraction)


def _line_key(text: str) -> str:
    return " ".join(_DIGITS.sub("#", text.lower()).split())


def _is_white(color: int) -> bool:
    return all(((color >> shift) & 0xFF) >= 0xF0 for shift in (16, 8, 0))


def _luminance(rgb: Sequence[float]) -> float:
    r, g, b = [*rgb, 1.0, 1.0, 1.0][:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _is_hidden(span: dict[str, Any], page_rect: Any, backgrounds: list[Any]) -> bool:
    """White text with nothing dark behind it, sub-2pt text, transparent text, or text off the page."""
    bbox = fitz.Rect(span["bbox"])
    if not bbox.intersects(page_rect) or span.get("alpha", 255) == 0:
        return True
    if float(span.get("size", 12.0)) < HIDDEN_MAX_FONT_PT:
        return True
    if _is_white(int(span.get("color", 0))):
        center = fitz.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
        return not any(bg.contains(center) for bg in backgrounds)  # white on a dark box is visible
    return False


def _has_grid(drawings: list[dict[str, Any]]) -> bool:
    """Enough horizontal and vertical strokes (lines or rectangle edges) to form a ruled table."""
    horizontal = vertical = 0
    for drawing in drawings:
        for item in drawing.get("items", ()):
            if item[0] == "l":
                (x0, y0), (x1, y1) = (item[1].x, item[1].y), (item[2].x, item[2].y)
                if abs(y1 - y0) < 1 and abs(x1 - x0) > 10:
                    horizontal += 1
                elif abs(x1 - x0) < 1 and abs(y1 - y0) > 10:
                    vertical += 1
            elif item[0] == "re":
                rect = item[1]
                if rect.width > 10 and rect.height > 5:
                    horizontal += 2
                    vertical += 2
                elif rect.width > 10:  # thin rectangles drawn as rules
                    horizontal += 1
                elif rect.height > 10:
                    vertical += 1
        if horizontal >= MIN_GRID_LINES and vertical >= MIN_GRID_LINES:
            return True
    return False


def _table_markdown(words: list[Any], table: Any) -> str:
    """Cell text from the page's words, each assigned to the cell holding its centre point.

    `Table.extract()` drops the spaces of PDFs that position words without space characters, and
    clipping by cell rectangle picks up letters of the neighbouring row; word centres avoid both.
    Columns that are empty in every row (artefacts of merged cells) are dropped.
    """
    grid: list[list[list[str]]] = []
    cells: list[tuple[int, int, tuple[float, float, float, float]]] = []
    for r, row in enumerate(table.rows):
        grid.append([[] for _ in row.cells])
        for c, bbox in enumerate(row.cells):
            if bbox is not None:
                cells.append((r, c, tuple(bbox)))
    for x0, y0, x1, y1, word, *_ in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for r, c, (bx0, by0, bx1, by1) in cells:
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                grid[r][c].append(word)
                break
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    text_grid = [[" ".join(cell) for cell in r] + [""] * (width - len(r)) for r in grid]
    keep = [i for i in range(width) if any(r[i] for r in text_grid)]
    rows = [[r[i] for i in keep] for r in text_grid if any(r[i] for i in keep)]
    if len(rows) < 2 or len(keep) < 2:
        return ""
    return rows_to_markdown(rows)


def _read_page(page: Any) -> _Page:
    rect = page.rect
    out = _Page(number=page.number + 1 if page.number is not None else 1, height=rect.height or 1.0)
    table_rects: list[Any] = []
    drawings = page.get_drawings()
    backgrounds = [
        fitz.Rect(d["rect"])
        for d in drawings
        if d.get("fill") is not None and _luminance(d["fill"]) < DARK_FILL_LUMINANCE
    ]
    backgrounds.extend(fitz.Rect(info["bbox"]) for info in page.get_image_info())
    if _has_grid(drawings):  # find_tables costs 0.3–2 s a page; ruled tables need vector lines
        try:
            words = page.get_text("words")
            for table in page.find_tables().tables:
                markdown = _table_markdown(words, table)
                if markdown:
                    table_rects.append(fitz.Rect(table.bbox))
                    out.tables.append(_Table(y0=float(table.bbox[1]), markdown=markdown))
        except Exception as exc:  # table detection is best effort
            log.debug("pdf.find_tables_failed", page=out.number, error=type(exc).__name__)

    data = page.get_text("dict")
    for block_no, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            parts: list[str] = []
            sizes: list[tuple[float, int]] = []
            bold = True
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    parts.append(text)
                    continue
                if _is_hidden(span, rect, backgrounds):
                    out.hidden += 1
                    continue
                parts.append(text)
                sizes.append((float(span.get("size", 0.0)), len(text)))
                bold = bold and bool(int(span.get("flags", 0)) & 16)
            text = " ".join("".join(parts).split())
            if not text or not sizes:
                continue
            x0, y0, x1, y1 = line["bbox"]
            center = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(r.contains(center) for r in table_rects):
                continue
            size = max(s for s, _ in sizes)
            out.lines.append(_Line(text, x0, y0, x1, y1, size, bold, block_no))
    return out


def _repeated_margin_keys(pages: list[_Page]) -> set[str]:
    if len(pages) < 2:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_line_key(line.text) for line in page.lines if page.in_margin(line)})
    threshold = max(2, int(len(pages) * REPEAT_PAGE_FRACTION + 0.5))
    return {key for key, n in counts.items() if n >= threshold}


def _header_date(pages: list[_Page]) -> date | None:
    """The in-document date, read from the page header of the first pages, then their body."""
    first = pages[:3]
    header_text = "\n".join(
        line.text for p in first for line in p.lines if line.y1 <= p.height * HEADER_DATE_FRACTION
    )
    if found := find_effective_date(header_text):
        return found
    body = "\n".join(line.text for p in first for line in p.lines)
    return find_effective_date(body)


def _body_font_size(pages: list[_Page]) -> float:
    weighted: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            weighted[round(line.size, 1)] += len(line.text)
    if not weighted:
        return 11.0
    return float(statistics.median(weighted.elements())) if len(weighted) > 1 else next(iter(weighted))


def _heading_level(line: _Line, body_size: float) -> int | None:
    text = line.text
    if not (2 <= len(text) <= 120) or not any(c.isalpha() for c in text) or text.endswith((".", ",", ";")):
        return None
    if line.size >= body_size * 1.6:
        return 1
    if line.size >= body_size * 1.25:
        return 2
    if line.bold and line.size >= body_size * 0.95 and len(text) <= 80:
        return 3
    return None


_NOT_A_TITLE = re.compile(r"^(table of )?contents$|^index$|^page \d+", re.IGNORECASE)


def _cover_title(page: _Page) -> str | None:
    """The largest-font line on the first page (before header stripping), if it reads like a title."""
    for line in sorted(page.lines, key=lambda ln: -ln.size):
        text = line.text.strip()
        if 8 <= len(text) <= 150 and any(c.isalpha() for c in text) and not _NOT_A_TITLE.match(text):
            return text
    return None


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return out


class PdfExtractor:
    file_types = frozenset({"pdf"})

    def __init__(
        self,
        *,
        ocr: PageOcr | None = None,
        min_script_ratio: float = 0.6,
        min_dict_hit: float = 0.15,
    ) -> None:
        self.ocr: PageOcr = ocr or NoOcr()
        self.min_script_ratio = min_script_ratio
        self.min_dict_hit = min_dict_hit

    def extract(self, path: Path) -> ExtractedDocument:
        try:
            doc = fitz.open(stream=path.read_bytes(), filetype="pdf")
        except Exception as exc:
            raise AppError(ErrorCode.CORRUPT_FILE, "The PDF could not be read.") from exc
        with doc:
            if doc.needs_pass:
                raise AppError(ErrorCode.ENCRYPTED_DOCUMENT, "The PDF is password-protected.")
            pages: list[_Page] = []
            for page in doc:
                try:
                    pages.append(_read_page(page))
                except Exception as exc:
                    log.warning("pdf.page_unreadable", page=(page.number or 0) + 1, error=type(exc).__name__)
                    pages.append(_Page(number=(page.number or 0) + 1, height=1.0))
            effective = _header_date(pages)  # before the header lines are stripped
            repeated = _repeated_margin_keys(pages)
            body_size = _body_font_size(pages)
            ocr_ready = self.ocr.available()

            # Pass 1: strip headers/footers and judge every page.
            kept_lines: list[list[_Line]] = []
            reports: list[QualityReport] = []
            for page in pages:
                kept = [
                    line
                    for line in page.lines
                    if not (
                        page.in_margin(line)
                        and (_line_key(line.text) in repeated or _PAGE_NUMBER.match(line.text))
                    )
                ]
                page_text = "\n".join([*(line.text for line in kept), *(t.markdown for t in page.tables)])
                kept_lines.append(kept)
                reports.append(
                    assess(page_text, min_script_ratio=self.min_script_ratio, min_dict_hit=self.min_dict_hit)
                )
            # A page too short to judge on its own is junk too when most judged pages are junk
            # (the short header-only pages of the scrambled NADRA Urdu PDF pass every check alone).
            judged = [r for r in reports if r.judged]
            junk_document = bool(judged) and (
                sum(1 for r in judged if r.reason in JUNK_REASONS) / len(judged) >= JUNK_DOCUMENT_FRACTION
            )
            if junk_document:
                reports = [
                    r
                    if r.judged or (not r.ok and r.reason != "too_short")
                    else replace(r, ok=False, reason="junk_document")
                    for r in reports
                ]

            # Pass 2: blocks, OCR fallback, unusable pages.
            tracker = SectionTracker()
            blocks: list[ExtractedBlock] = []
            ocr_pages = 0
            unusable: list[int] = []
            quality: dict[int, str] = {}
            title: str | None = _cover_title(pages[0]) if pages and reports[0].ok else None
            for page, kept, report in zip(pages, kept_lines, reports, strict=True):
                if not report.ok and ocr_ready:
                    ocr_blocks = self._ocr(doc[page.number - 1], page.number, tracker.path)
                    if ocr_blocks:
                        blocks.extend(ocr_blocks)
                        ocr_pages += 1
                        quality[page.number] = f"ocr:{report.reason}"
                        continue
                if not report.ok and report.reason != "too_short":
                    unusable.append(page.number)  # junk text layer and no usable OCR: never indexed
                    quality[page.number] = f"unusable:{report.reason}"
                    continue
                page_blocks, page_title = self._layout_blocks(page, kept, body_size, tracker)
                title = title or page_title
                blocks.extend(page_blocks)

            meta_title = (doc.metadata or {}).get("title") or ""
            if len(meta_title.strip()) > 3 and not re.search(r"\.(docx?|pdf)$|^microsoft", meta_title, re.I):
                title = meta_title.strip()
            metadata: dict[str, Any] = {"unusable_pages": unusable, "page_quality": quality}
            if effective:
                metadata["effective_date"] = effective
            return ExtractedDocument(
                blocks=tuple(blocks),
                page_count=len(pages),
                title=title,
                hidden_text_removed=sum(p.hidden for p in pages),
                ocr_pages=ocr_pages,
                metadata=metadata,
            )

    def _ocr(self, page: Any, number: int, section: tuple[str, ...]) -> list[ExtractedBlock]:
        try:
            result = self.ocr.ocr_page(page)
        except Exception as exc:
            log.warning("pdf.ocr_failed", page=number, error=type(exc).__name__)
            return []
        report = assess(result.text, min_script_ratio=self.min_script_ratio, min_dict_hit=self.min_dict_hit)
        if not report.ok and report.reason != "low_dict_hit":
            return []
        return [
            ExtractedBlock(
                text=para.strip(),
                kind="ocr",
                page=number,
                section_path=section,
                is_ocr=True,
                ocr_confidence=result.confidence,
            )
            for para in result.text.split("\n\n")
            if para.strip()
        ]

    @staticmethod
    def _layout_blocks(
        page: _Page, lines: list[_Line], body_size: float, tracker: SectionTracker
    ) -> tuple[list[ExtractedBlock], str | None]:
        items: list[tuple[float, ExtractedBlock]] = []
        title: str | None = None
        para: list[str] = []
        para_y = 0.0
        para_block = -1

        def flush() -> None:
            nonlocal para
            if para:
                items.append(
                    (
                        para_y,
                        ExtractedBlock(
                            text=_join_lines(para),
                            kind="paragraph",
                            page=page.number,
                            section_path=tracker.path,
                        ),
                    )
                )
                para = []

        for line in lines:
            level = _heading_level(line, body_size)
            if level is not None:
                flush()
                path = tracker.enter(level, line.text)
                if level == 1 and title is None and page.number <= 2 and not _NOT_A_TITLE.match(line.text):
                    title = line.text
                items.append(
                    (
                        line.y0,
                        ExtractedBlock(text=line.text, kind="heading", page=page.number, section_path=path),
                    )
                )
                continue
            if line.block_no != para_block:
                flush()
                para_block = line.block_no
                para_y = line.y0
            para.append(line.text)
        flush()

        blocks = [b for _, b in items]
        # Insert each table before the first text item that starts below it (reading order).
        for table in sorted(page.tables, key=lambda t: t.y0):
            pos = next((i for i, (y, _) in enumerate(items) if y >= table.y0), len(items))
            section = items[pos - 1][1].section_path if pos > 0 else tracker.path
            block = ExtractedBlock(text=table.markdown, kind="table", page=page.number, section_path=section)
            items.insert(pos, (table.y0, block))
            blocks = [b for _, b in items]
        return blocks, title
