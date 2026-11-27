"""Measure Urdu/English text quality on real corpus files before ingestion (feedback item 3, E46).

Usage:
    .venv\\Scripts\\python scripts\\ocr_check.py <file.pdf|file.png|file.jpg> [more files] [--pages 1-3]

For every page it reports:
  - text_layer: how much text PyMuPDF extracts, the Arabic-script ratio, and an Urdu/English
    common-word hit rate (a cheap "is this real text or junk" signal);
  - ocr (only if Tesseract with `urd` + `eng` is installed): the same numbers on OCR output of the
    rendered page, plus Tesseract's mean confidence.
The verdict per page is `text_ok`, `use_ocr` or `unusable`. The thresholds are the ones in
Settings (TEXT_QUALITY_MIN_SCRIPT_RATIO, TEXT_QUALITY_MIN_DICT_HIT), so this script and the
ingestion pipeline agree.

Needs:  uv pip install pymupdf pytesseract pillow   (and Tesseract itself for the OCR columns).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings

URDU_COMMON = {
    "کے", "کی", "کا", "میں", "ہے", "اور", "سے", "کو", "پر", "یہ", "ہیں", "کہ", "نہیں", "کیا", "جو",
    "گا", "گی", "ایک", "لیے", "بھی", "شناختی", "کارڈ", "درخواست", "فیس", "حکومت", "قانون",
}  # fmt: skip
ENGLISH_COMMON = {
    "the", "of", "and", "to", "in", "is", "for", "be", "by", "or", "with", "on", "as", "shall",
    "any", "not", "fee", "card", "application", "act", "section", "government", "rules",
}  # fmt: skip
ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
WORD = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


def quality(text: str) -> dict[str, float]:
    text = unicodedata.normalize("NFKC", text)  # Arabic presentation forms -> base letters
    letters = LETTER.findall(text)
    arabic = ARABIC.findall(text)
    words = WORD.findall(text.lower())
    hits = sum(1 for w in words if w in URDU_COMMON or w in ENGLISH_COMMON)
    return {
        "chars": float(len(text.strip())),
        "arabic_ratio": round(len(arabic) / len(letters), 3) if letters else 0.0,
        "dict_hit": round(hits / len(words), 3) if words else 0.0,
    }


def verdict(q: dict[str, float], expect_urdu: bool, s: Settings) -> str:
    if q["chars"] < 25:
        return "unusable"
    script_ok = q["arabic_ratio"] >= s.text_quality_min_script_ratio if expect_urdu else True
    return "text_ok" if script_ok and q["dict_hit"] >= s.text_quality_min_dict_hit else "use_ocr"


def parse_pages(spec: str | None, count: int) -> list[int]:
    if not spec:
        return list(range(min(count, 3)))
    a, _, b = spec.partition("-")
    return list(range(int(a) - 1, min(int(b or a), count)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--pages", help="e.g. 1-3 (default: first 3 pages)")
    ap.add_argument("--urdu", action="store_true", help="the document is expected to be Urdu")
    args = ap.parse_args()

    try:
        import pymupdf as fitz
    except ImportError:
        print("Install PyMuPDF first:  uv pip install --python .venv pymupdf pytesseract pillow")
        return 2

    settings = Settings()
    tesseract = shutil.which("tesseract")
    ocr = None
    if tesseract:
        import pytesseract
        from PIL import Image

        langs = pytesseract.get_languages(config="")
        if "urd" not in langs:
            print("Tesseract found but the 'urd' language data is missing; OCR columns are skipped.")
        else:

            def ocr(img: Image.Image) -> tuple[str, float]:
                data = pytesseract.image_to_data(
                    img, lang=settings.ocr_langs, output_type=pytesseract.Output.DICT
                )
                confs = [float(c) for c in data["conf"] if str(c) not in ("-1", "")]
                text = " ".join(w for w in data["text"] if w.strip())
                return text, round(sum(confs) / len(confs), 1) if confs else 0.0
    else:
        print("Tesseract is not installed: only the text-layer columns are measured.")

    for path in args.files:
        print(f"\n=== {path.name}")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if not ocr:
                print("  image file: needs Tesseract with 'urd' to measure")
                continue
            from PIL import Image

            text, conf = ocr(Image.open(path))
            q = quality(text)
            print(f"  ocr: {q} conf={conf} -> {verdict(q, args.urdu, settings)}")
            continue
        with fitz.open(path) as doc:
            for i in parse_pages(args.pages, doc.page_count):
                page = doc[i]
                q = quality(page.get_text())
                line = f"  p{i + 1}: text_layer={q} -> {verdict(q, args.urdu, settings)}"
                if ocr:
                    from PIL import Image

                    pix = page.get_pixmap(dpi=300)  # render, so vector-outline text is caught too
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    text, conf = ocr(img)
                    oq = quality(text)
                    line += f"\n       ocr={oq} conf={conf} -> {verdict(oq, args.urdu, settings)}"
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
