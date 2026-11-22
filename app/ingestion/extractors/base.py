"""Helpers shared by the extractors."""

from __future__ import annotations

from collections.abc import Sequence

from charset_normalizer import from_bytes

from app.core.errors import AppError, ErrorCode
from app.protocols import ExtractedBlock


def rows_to_markdown(rows: Sequence[Sequence[object | None]]) -> str:
    """A list of rows (first row = header) → a GitHub Markdown table. Empty rows are dropped."""
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [" ".join(str(c).split()) if c is not None else "" for c in row]
        if any(cells):
            cleaned.append([c.replace("|", "\\|") for c in cells])
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def decode_text(raw: bytes, what: str = "text file") -> str:
    """Decode bytes with charset detection (UTF-8/16, cp1256 for legacy Urdu files, …)."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    best = from_bytes(raw).best()
    if best is None:
        raise AppError(ErrorCode.CORRUPT_FILE, f"The {what} could not be decoded.")
    return str(best)


class SectionTracker:
    """Keeps the heading path (`section_path`) while walking a document in order."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def enter(self, level: int, title: str) -> tuple[str, ...]:
        title = " ".join(title.split())
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        if title:
            self._stack.append((level, title[:200]))
        return self.path

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(t for _, t in self._stack)


def dedupe_blocks(blocks: Sequence[ExtractedBlock], min_chars: int = 40) -> list[ExtractedBlock]:
    """Drop repeated non-heading blocks (responsive duplicates, repeated tab panes in web pages)."""
    seen: set[str] = set()
    out: list[ExtractedBlock] = []
    for b in blocks:
        key = " ".join(b.text.split())
        if b.kind != "heading" and len(key) >= min_chars:
            if key in seen:
                continue
            seen.add(key)
        out.append(b)
    return out
