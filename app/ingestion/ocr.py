"""OCR of rendered PDF pages with Tesseract `eng+urd` (blueprint §3.2, E1, E46).

The *rendered page* is OCR'd (not the embedded images), so text drawn as vector outlines is caught
too. Tesseract is optional: when the binary or the `urd` language data is missing,
`TesseractOcr.available()` is False and failing pages are marked unusable instead.
"""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)

OCR_DPI = 300


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    confidence: float | None  # mean word confidence, 0–100


@runtime_checkable
class PageOcr(Protocol):
    def available(self) -> bool: ...

    def ocr_page(self, page: Any) -> OcrResult: ...


class NoOcr:
    """OCR switched off (Settings.ocr_enabled = False)."""

    def available(self) -> bool:
        return False

    def ocr_page(self, page: Any) -> OcrResult:
        raise RuntimeError("OCR is disabled")


class TesseractOcr:
    def __init__(self, langs: str = "eng+urd", dpi: int = OCR_DPI) -> None:
        self.langs = langs
        self.dpi = dpi
        self._available: bool | None = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        with self._lock:
            if self._available is None:
                self._available = self._probe()
            return self._available

    def _probe(self) -> bool:
        binary = shutil.which("tesseract")
        if not binary:
            log.info("ocr.tesseract_missing")
            return False
        try:
            import pytesseract  # type: ignore[import-untyped]

            pytesseract.pytesseract.tesseract_cmd = binary
            installed = set(pytesseract.get_languages(config=""))
        except Exception as exc:
            log.warning("ocr.tesseract_probe_failed", error=type(exc).__name__)
            return False
        missing = set(self.langs.split("+")) - installed
        if missing:
            log.warning("ocr.languages_missing", missing=sorted(missing))
            return False
        return True

    def ocr_page(self, page: Any) -> OcrResult:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=self.dpi)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        data = pytesseract.image_to_data(image, lang=self.langs, output_type=pytesseract.Output.DICT)
        lines: dict[tuple[int, int, int], list[str]] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            if not str(word).strip():
                continue
            key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            lines.setdefault(key, []).append(str(word))
            conf = float(data["conf"][i])
            if conf >= 0:
                confs.append(conf)
        paragraphs: dict[tuple[int, int], list[str]] = {}
        for (block, par, _line), words in sorted(lines.items()):
            paragraphs.setdefault((block, par), []).append(" ".join(words))
        text = "\n\n".join("\n".join(ls) for _, ls in sorted(paragraphs.items()))
        return OcrResult(text=text, confidence=round(sum(confs) / len(confs), 1) if confs else None)
