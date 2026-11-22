"""Upload validation: document guardrails GD-1 to GD-8 (blueprint §7.3).

`validate_file` runs before any parser touches the file and returns the canonical file type.
Every rejection is an `AppError` with a stable code:

| check | code |
|---|---|
| extension not allowed, or magic bytes disagree with it | 415 `unsupported_media_type` |
| 0 bytes | 400 `empty_file` |
| above the upload cap / page cap | 413 `file_too_large` |
| PDF needs a password | 400 `encrypted_document` |
| parser or zip errors, XML attacks (XXE) | 400 `corrupt_file` |
| zip bomb (> 200 MB uncompressed or ratio > 100) | 400 `malicious_archive` |
| macros (`.docm`, `vbaProject.bin`); PDF `/JavaScript`, `/JS`, `/Launch`, `/EmbeddedFile` | 400 |

(the last row's code is `active_content_rejected`)

HTML `<script>` is not rejected: the HTML extractor strips it (it is never executed).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePath
from typing import Any

import defusedxml.ElementTree as SafeET  # type: ignore[import-untyped]
import filetype  # type: ignore[import-untyped]
from defusedxml import DefusedXmlException

from app.config import Settings
from app.core.errors import AppError, ErrorCode

ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
}
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})  # D4 stretch: OCR path not enabled yet
MACRO_EXTENSIONS = frozenset({".docm", ".dotm", ".pptm", ".potm", ".ppsm", ".xlsm", ".xltm"})
TEXT_TYPES = frozenset({"html", "txt", "md"})
OOXML_MAIN_PART = {"docx": "word/document.xml", "pptx": "ppt/presentation.xml"}

ZIP_MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
ZIP_MAX_RATIO = 100.0
XML_PARSE_MAX_BYTES = 50 * 1024 * 1024  # parts above this are not pre-parsed (already bomb-checked)
TEXT_SNIFF_BYTES = 64 * 1024

_PDF_ACTIVE = re.compile(r"/(JavaScript|JS|Launch|EmbeddedFiles?|RichMedia)(?![A-Za-z0-9])")
_MACRO_CONTENT_TYPE = re.compile(r"macroEnabled|vbaProject|activeX", re.IGNORECASE)


def _reject(code: ErrorCode, detail: str, **extra: object) -> AppError:
    return AppError(code, detail, extra=dict(extra))


def file_extension(file_name: str) -> str:
    return PurePath(file_name).suffix.lower()


def type_from_name(file_name: str) -> str:
    """Canonical type for an allowed extension; raises 415 / 400 for everything else."""
    ext = file_extension(file_name)
    if ext in MACRO_EXTENSIONS:
        raise _reject(
            ErrorCode.ACTIVE_CONTENT_REJECTED, "Macro-enabled Office files are not accepted.", extension=ext
        )
    if ext in IMAGE_EXTENSIONS:
        raise _reject(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE, "Image uploads (JPG/PNG) are not enabled yet.", extension=ext
        )
    if ext not in ALLOWED_EXTENSIONS:
        raise _reject(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            "Unsupported file type. Allowed: PDF, DOCX, PPTX, HTML, TXT, MD.",
            extension=ext or None,
        )
    return ALLOWED_EXTENSIONS[ext]


def _looks_like_text(head: bytes) -> bool:
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 with BOM
        return True
    if b"\x00" in head:
        return False
    from charset_normalizer import from_bytes

    return from_bytes(head).best() is not None


def _sniff(path: Path, declared: str) -> None:
    """The magic bytes must agree with the extension (GD-1, E18)."""
    kind = filetype.guess(str(path))
    guessed: str | None = kind.extension if kind is not None else None
    with path.open("rb") as f:
        head = f.read(TEXT_SNIFF_BYTES)
    if declared == "pdf":
        if guessed == "pdf" or head.lstrip()[:5] == b"%PDF-":
            return
        if guessed is None and _looks_like_text(head[:4096]) is False:
            raise _reject(ErrorCode.CORRUPT_FILE, "The PDF could not be read.")
    elif declared in OOXML_MAIN_PART:
        if guessed in {declared, "zip"} or head[:4] == b"PK\x03\x04":
            return
        if guessed is None:
            raise _reject(ErrorCode.CORRUPT_FILE, f"The {declared.upper()} file could not be read.")
    elif declared in TEXT_TYPES and guessed is None and _looks_like_text(head):
        return
    raise _reject(
        ErrorCode.UNSUPPORTED_MEDIA_TYPE,
        "The file content does not match its extension.",
        declared=declared,
        detected=guessed,
    )


def check_ooxml(path: Path, declared: str) -> None:
    """Zip-bomb, macro and XML-attack checks for DOCX/PPTX (GD-6, GD-7, GD-8)."""
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            names = {i.filename for i in infos}
            total = sum(i.file_size for i in infos)
            compressed = sum(i.compress_size for i in infos) or 1
            if total > ZIP_MAX_UNCOMPRESSED_BYTES or total / compressed > ZIP_MAX_RATIO:
                raise _reject(
                    ErrorCode.MALICIOUS_ARCHIVE,
                    "The archive expands to an unsafe size.",
                    uncompressed_bytes=total,
                    ratio=round(total / compressed, 1),
                )
            if any(PurePath(n).name.lower() in {"vbaproject.bin", "vbadata.xml"} for n in names) or any(
                n.lower().startswith(("word/activex/", "ppt/activex/")) for n in names
            ):
                raise _reject(ErrorCode.ACTIVE_CONTENT_REJECTED, "The document contains macros.")
            main = OOXML_MAIN_PART[declared]
            if "[Content_Types].xml" not in names or main not in names:
                raise _reject(
                    ErrorCode.UNSUPPORTED_MEDIA_TYPE,
                    f"The file is a zip archive but not a {declared.upper()} document.",
                )
            content_types = zf.read("[Content_Types].xml")
            if _MACRO_CONTENT_TYPE.search(content_types.decode("utf-8", errors="replace")):
                raise _reject(ErrorCode.ACTIVE_CONTENT_REJECTED, "The document is macro-enabled.")
            for info in infos:
                if info.filename.endswith((".xml", ".rels")) and info.file_size <= XML_PARSE_MAX_BYTES:
                    SafeET.fromstring(zf.read(info))  # raises on DTDs, entities, external refs
    except AppError:
        raise
    except DefusedXmlException as exc:
        raise _reject(ErrorCode.CORRUPT_FILE, "The document contains unsafe XML.") from exc
    except (zipfile.BadZipFile, zipfile.LargeZipFile, SafeET.ParseError, KeyError, OSError, EOFError) as exc:
        raise _reject(ErrorCode.CORRUPT_FILE, f"The {declared.upper()} file is corrupt.") from exc


def check_pdf(path: Path, settings: Settings) -> None:
    """Encrypted, corrupt, oversized and active-content checks for PDFs (GD-4, GD-5, GD-7)."""
    import pymupdf

    fitz: Any = pymupdf

    try:
        doc = fitz.open(stream=path.read_bytes(), filetype="pdf")  # no OS file handle left open (Windows)
    except Exception as exc:  # pymupdf raises FileDataError / RuntimeError subclasses
        raise _reject(ErrorCode.CORRUPT_FILE, "The PDF could not be read.") from exc
    with doc:
        if doc.needs_pass:
            raise _reject(ErrorCode.ENCRYPTED_DOCUMENT, "The PDF is password-protected.")
        if doc.page_count < 1:
            raise _reject(ErrorCode.CORRUPT_FILE, "The PDF has no pages.")
        if doc.page_count > settings.max_pages:
            raise _reject(
                ErrorCode.FILE_TOO_LARGE,
                f"The PDF has {doc.page_count} pages; the limit is {settings.max_pages}.",
                pages=doc.page_count,
                limit=settings.max_pages,
            )
        if doc.embfile_count() > 0:
            raise _reject(ErrorCode.ACTIVE_CONTENT_REJECTED, "The PDF contains embedded files.")
        try:
            for xref in range(1, doc.xref_length()):
                obj = doc.xref_object(xref, compressed=True)
                if (m := _PDF_ACTIVE.search(obj)) is not None:
                    raise _reject(
                        ErrorCode.ACTIVE_CONTENT_REJECTED,
                        "The PDF contains active content.",
                        feature=m.group(1),
                    )
        except AppError:
            raise
        except Exception as exc:
            raise _reject(ErrorCode.CORRUPT_FILE, "The PDF structure is damaged.") from exc


def validate_file(path: Path, file_name: str, settings: Settings) -> str:
    """Run every pre-parse check and return the canonical file type (`pdf`, `docx`, …)."""
    declared = type_from_name(file_name)
    size = path.stat().st_size
    if size == 0:
        raise _reject(ErrorCode.EMPTY_FILE, "The file is empty.")
    if size > settings.max_upload_bytes:
        raise _reject(
            ErrorCode.FILE_TOO_LARGE,
            f"The file is larger than {settings.max_upload_mb} MB.",
            limit_bytes=settings.max_upload_bytes,
        )
    _sniff(path, declared)
    if declared == "pdf":
        check_pdf(path, settings)
    elif declared in OOXML_MAIN_PART:
        check_ooxml(path, declared)
    return declared
