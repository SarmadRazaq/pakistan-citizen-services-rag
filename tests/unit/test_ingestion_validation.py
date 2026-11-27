"""Document guardrails GD-1..GD-8 (blueprint §7.3): E2, E5, E18–E21."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.ingestion.validation import validate_file
from tests.fixtures.files import factories as f


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, app_env="test", max_upload_mb=1, max_pages=5)  # type: ignore[call-arg]


def _code(path: Path, name: str, settings: Settings) -> ErrorCode:
    with pytest.raises(AppError) as info:
        validate_file(path, name, settings)
    return info.value.code


def test_valid_files_return_their_type(tmp_path: Path, settings: Settings) -> None:
    pdf = f.make_pdf(tmp_path / "a.pdf", [[f.ENGLISH_PARAGRAPH]])
    docx_file = f.make_docx(tmp_path / "a.docx")
    pptx_file = f.make_pptx(tmp_path / "a.pptx", [("Intro", "Hello")])
    html = tmp_path / "a.html"
    html.write_text("<html><body><p>Hello</p></body></html>", encoding="utf-8")
    txt = tmp_path / "a.txt"
    txt.write_text("نادرا شناختی کارڈ کی فیس", encoding="utf-8")
    md = tmp_path / "a.md"
    md.write_text("# Title\n\nBody", encoding="utf-8")
    assert validate_file(pdf, "Guide.PDF", settings) == "pdf"
    assert validate_file(docx_file, "a.docx", settings) == "docx"
    assert validate_file(pptx_file, "a.pptx", settings) == "pptx"
    assert validate_file(html, "page.htm", settings) == "html"
    assert validate_file(txt, "a.txt", settings) == "txt"
    assert validate_file(md, "a.md", settings) == "md"


def test_empty_file(tmp_path: Path, settings: Settings) -> None:  # E2
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    assert _code(empty, "empty.pdf", settings) is ErrorCode.EMPTY_FILE


def test_too_large(tmp_path: Path, settings: Settings) -> None:  # E5
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (settings.max_upload_bytes + 1))
    assert _code(big, "big.txt", settings) is ErrorCode.FILE_TOO_LARGE


def test_page_cap(tmp_path: Path, settings: Settings) -> None:  # E6: page cap
    pdf = f.make_pdf(tmp_path / "long.pdf", [["page"]] * 6)
    assert _code(pdf, "long.pdf", settings) is ErrorCode.FILE_TOO_LARGE


@pytest.mark.parametrize("name", ["virus.exe", "photo.jpg", "scan.png", "data.xlsx", "noext"])
def test_unsupported_extensions(tmp_path: Path, settings: Settings, name: str) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"hello world")
    assert _code(path, name, settings) is ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_spoofed_extension(tmp_path: Path, settings: Settings) -> None:  # E18
    exe = tmp_path / "invoice.pdf"
    exe.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff" + b"\x00" * 200)
    assert _code(exe, "invoice.pdf", settings) is ErrorCode.UNSUPPORTED_MEDIA_TYPE
    png = tmp_path / "notes.txt"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert _code(png, "notes.txt", settings) is ErrorCode.UNSUPPORTED_MEDIA_TYPE
    pdf_as_docx = f.make_pdf(tmp_path / "x.docx", [["hi"]])
    assert _code(pdf_as_docx, "x.docx", settings) is ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_plain_zip_is_not_a_docx(tmp_path: Path, settings: Settings) -> None:
    path = f.add_zip_member(tmp_path / "a.docx", "readme.txt", b"hello")
    assert _code(path, "a.docx", settings) is ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_encrypted_pdf(tmp_path: Path, settings: Settings) -> None:  # E19
    assert _code(f.make_encrypted_pdf(tmp_path / "e.pdf"), "e.pdf", settings) is ErrorCode.ENCRYPTED_DOCUMENT


def test_corrupt_files(tmp_path: Path, settings: Settings) -> None:  # E20
    pdf = tmp_path / "c.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj garbage garbage\n%%EOF")
    assert _code(pdf, "c.pdf", settings) is ErrorCode.CORRUPT_FILE
    random_pdf = tmp_path / "r.pdf"
    random_pdf.write_bytes(bytes(range(256)) * 4)
    assert _code(random_pdf, "r.pdf", settings) is ErrorCode.CORRUPT_FILE
    truncated = f.make_docx(tmp_path / "t.docx")
    truncated.write_bytes(truncated.read_bytes()[:300])
    assert _code(truncated, "t.docx", settings) is ErrorCode.CORRUPT_FILE
    junk = tmp_path / "j.docx"
    junk.write_bytes(bytes(range(256)) * 4)
    assert _code(junk, "j.docx", settings) is ErrorCode.CORRUPT_FILE


def test_zip_bomb(tmp_path: Path) -> None:  # E21
    roomy = Settings(_env_file=None, app_env="test", max_upload_mb=50)  # type: ignore[call-arg]
    bomb = f.make_zip_bomb_docx(tmp_path / "bomb.docx", megabytes=20)
    assert _code(bomb, "bomb.docx", roomy) is ErrorCode.MALICIOUS_ARCHIVE


def test_macro_documents(tmp_path: Path, settings: Settings) -> None:  # E21
    docm = f.make_docx(tmp_path / "m.docm")
    assert _code(docm, "m.docm", settings) is ErrorCode.ACTIVE_CONTENT_REJECTED
    vba = f.make_macro_docx(tmp_path / "vba.docx")
    assert _code(vba, "vba.docx", settings) is ErrorCode.ACTIVE_CONTENT_REJECTED


def test_xxe_in_ooxml_is_rejected(tmp_path: Path, settings: Settings) -> None:  # GD-8
    assert _code(f.make_xxe_docx(tmp_path / "x.docx"), "x.docx", settings) is ErrorCode.CORRUPT_FILE


@pytest.mark.parametrize("builder", [f.make_js_pdf, f.make_embedded_file_pdf])
def test_pdf_active_content(tmp_path: Path, settings: Settings, builder: object) -> None:  # GD-7
    path = builder(tmp_path / "a.pdf")  # type: ignore[operator]
    assert _code(path, "a.pdf", settings) is ErrorCode.ACTIVE_CONTENT_REJECTED
