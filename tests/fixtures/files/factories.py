"""Builders for tiny test documents (PDF via PyMuPDF, DOCX via python-docx, PPTX via python-pptx).

Every builder writes into a caller-supplied path (usually under `tmp_path`) and returns it.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import docx
import pymupdf
from docx.shared import Pt, RGBColor
from pptx import Presentation
from pptx.util import Inches

fitz: Any = pymupdf

ENGLISH_PARAGRAPH = (
    "The applicant shall submit the application form at the registration office of the authority. "
    "A fee is charged for the card and the processing time depends on the category that is selected. "
    "The office will issue the card to the person after the verification of the documents."
)
GARBAGE_LATIN = (
    "xqzv kjtr wpqm ztrl vbnq kxjw qpzm trvx lmqz wkjp zxqv rtpl mnbv qwzx plkj vtrz qmxn jzkw "
    "xvbt rqlp zmwk tjxq vplr kzqm wxjt qrvz lpmk zjtx vqwr mkpl xztj rwqv plzm"
)


def make_pdf(
    path: Path,
    pages: Sequence[Sequence[str]],
    *,
    header: str | None = None,
    footer_page_numbers: bool = False,
    fontsize: float = 11,
) -> Path:
    """One list of paragraphs per page; optional repeated header and "Page N of M" footer."""
    doc = fitz.open()
    total = len(pages)
    for number, paragraphs in enumerate(pages, start=1):
        page = doc.new_page()  # 595 x 842
        if header:
            page.insert_text((72, 30), header, fontsize=9)
        y = 110.0
        for para in paragraphs:
            rect = fitz.Rect(72, y, 523, y + 220)
            page.insert_textbox(rect, para, fontsize=fontsize)
            y += 120
        if footer_page_numbers:
            page.insert_text((280, 820), f"Page {number} of {total}", fontsize=9)
    doc.save(path)
    doc.close()
    return path


def make_pdf_with_hidden_text(path: Path, visible: str, hidden: str) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(72, 100, 523, 300), visible, fontsize=11)
    page.insert_text((72, 400), hidden, fontsize=11, color=(1, 1, 1))  # white on white
    page.insert_text((72, 420), "tiny " + hidden, fontsize=1)  # 1 pt
    # White text on a dark box is visible and must be kept.
    page.draw_rect(fitz.Rect(70, 450, 400, 475), color=(0, 0, 0.5), fill=(0, 0, 0.5))
    page.insert_text((72, 468), "Banner heading on a dark box", fontsize=11, color=(1, 1, 1))
    doc.save(path)
    doc.close()
    return path


def make_pdf_with_table(path: Path, intro: str, rows: Sequence[Sequence[str]]) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(72, 60, 523, 160), intro, fontsize=11)
    cells = "".join(
        "<tr>" + "".join(f"<td style='border:1px solid black;padding:2px'>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    page.insert_htmlbox(
        fitz.Rect(72, 200, 523, 600), f"<table style='border-collapse:collapse'>{cells}</table>"
    )
    doc.save(path)
    doc.close()
    return path


def make_scanned_pdf(path: Path, pages: int = 1) -> Path:
    """Pages with drawings but no text layer, like a scan without OCR."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.draw_rect(fitz.Rect(72, 72, 300, 200), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    doc.save(path)
    doc.close()
    return path


def make_encrypted_pdf(path: Path) -> Path:
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "secret", fontsize=11)
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="user-pw", owner_pw="owner-pw")
    doc.close()
    return path


def make_js_pdf(path: Path) -> Path:
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(72, 100, 523, 300), ENGLISH_PARAGRAPH, fontsize=11)
    doc.xref_set_key(doc.pdf_catalog(), "OpenAction", "<</S/JavaScript/JS(app.alert\\(1\\))>>")
    doc.save(path)
    doc.close()
    return path


def make_embedded_file_pdf(path: Path) -> Path:
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(72, 100, 523, 300), ENGLISH_PARAGRAPH, fontsize=11)
    doc.embfile_add("payload.txt", b"not a document")
    doc.save(path)
    doc.close()
    return path


def make_docx(
    path: Path,
    *,
    title: str | None = "Passport Guide",
    sections: Sequence[tuple[str, Sequence[str]]] = (("Requirements", (ENGLISH_PARAGRAPH,)),),
    table: Sequence[Sequence[str]] | None = None,
    hidden: str | None = None,
) -> Path:
    document = docx.Document()
    if title:
        document.add_heading(title, level=1)
    for heading, paragraphs in sections:
        document.add_heading(heading, level=2)
        for para in paragraphs:
            document.add_paragraph(para)
    if table:
        t = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    if hidden:
        para = document.add_paragraph("Visible sentence. ")
        run = para.add_run(hidden)
        run.font.hidden = True
        white = para.add_run(" white " + hidden)
        white.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        tiny = para.add_run(" tiny " + hidden)
        tiny.font.size = Pt(1)
    document.save(str(path))
    return path


def add_zip_member(path: Path, name: str, data: bytes, *, compress: int = zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "a", compression=compress) as zf:
        zf.writestr(name, data)
    return path


def make_zip_bomb_docx(path: Path, megabytes: int = 30) -> Path:
    make_docx(path)
    return add_zip_member(path, "word/media/bomb.bin", b"\0" * (megabytes * 1024 * 1024))


def make_macro_docx(path: Path) -> Path:
    make_docx(path)
    return add_zip_member(path, "word/vbaProject.bin", b"\xd0\xcf\x11\xe0 fake vba")


def make_xxe_docx(path: Path) -> Path:
    make_docx(path)
    xml = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><x>&e;</x>'
    return add_zip_member(path, "word/attack.xml", xml)


def make_pptx(path: Path, slides: Sequence[tuple[str, str]]) -> Path:
    deck = Presentation()
    layout = deck.slide_layouts[1]
    for title, body in slides:
        slide = deck.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    table_slide = deck.slides.add_slide(deck.slide_layouts[5])
    table_slide.shapes.title.text = "Fees"
    shape = table_slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(6), Inches(1))
    for (r, c), value in {(0, 0): "Category", (0, 1): "Fee", (1, 0): "Normal", (1, 1): "Rs 1500"}.items():
        shape.table.cell(r, c).text = value
    deck.save(str(path))
    return path
