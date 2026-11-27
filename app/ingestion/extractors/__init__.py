"""Extractors per file type (blueprint §3.2). Each implements `app.protocols.Extractor`."""

from __future__ import annotations

from app.core.errors import AppError, ErrorCode
from app.ingestion.extractors.docx import DocxExtractor
from app.ingestion.extractors.html import HtmlExtractor
from app.ingestion.extractors.markdown import MarkdownExtractor
from app.ingestion.extractors.pdf import PdfExtractor
from app.ingestion.extractors.pptx import PptxExtractor
from app.ingestion.extractors.text import TextExtractor
from app.ingestion.ocr import PageOcr
from app.protocols import Extractor

__all__ = [
    "DocxExtractor",
    "HtmlExtractor",
    "MarkdownExtractor",
    "PdfExtractor",
    "PptxExtractor",
    "TextExtractor",
    "default_extractors",
    "extractor_for",
]


def default_extractors(
    *, ocr: PageOcr | None = None, min_script_ratio: float = 0.6, min_dict_hit: float = 0.15
) -> dict[str, Extractor]:
    extractors: list[Extractor] = [
        PdfExtractor(ocr=ocr, min_script_ratio=min_script_ratio, min_dict_hit=min_dict_hit),
        HtmlExtractor(),
        DocxExtractor(),
        PptxExtractor(),
        TextExtractor(),
        MarkdownExtractor(),
    ]
    return {file_type: e for e in extractors for file_type in e.file_types}


def extractor_for(extractors: dict[str, Extractor], file_type: str) -> Extractor:
    try:
        return extractors[file_type]
    except KeyError:
        raise AppError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, f"No extractor for '{file_type}' files.") from None
