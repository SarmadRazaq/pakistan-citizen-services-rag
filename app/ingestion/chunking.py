"""Structure-aware recursive chunking (blueprint §3.3).

- Hard boundaries are section changes (headings, slides). Within a section, text blocks are packed
  greedily up to `chunk_size` tokens with `overlap` tokens carried over from the previous chunk.
  A block that is too big is split recursively: paragraph → line → sentence (English and Urdu
  enders `. ! ? ۔ ؟`) → Urdu comma `،` → words → characters.
- Every text chunk is an exact slice of the document text: `full_text[char_start:char_end]`.
  That makes "no text lost" and "positions are monotonic" checkable (see the property test).
- Tables are atomic. A table larger than the limit is split by rows, repeating the header row.
- Sizes are measured with a `TokenCounter`. The real one is the embedding model's tokenizer
  (BGE-M3 / XLM-R `tokenizer.json`, loaded lazily); tests use `RegexTokenCounter` and never download.
- The embedding text gets a context header `{agency} ({jurisdiction}) › {title} › {section}`;
  the stored display text stays clean.
"""

from __future__ import annotations

import itertools
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import structlog

from app.ingestion.cleaning import join_blocks
from app.ingestion.language import detect_language, detect_script
from app.protocols import ExtractedBlock

log = structlog.get_logger(__name__)

SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "۔ ",
    ". ",
    "؟ ",
    "? ",
    "! ",
    "۔",
    "؟",
    "، ",
    "،",
    "; ",
    ", ",
    " ",
)
_FINE_LEVELS = SEPARATORS.index("؟") + 1  # paragraph, line and sentence separators
_TABLE_SEPARATOR_ROW = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")


# ---------------------------------------------------------------- token counting


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class RegexTokenCounter:
    """Words and punctuation marks. Deterministic, offline; used by tests and as a fallback."""

    _TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        return len(self._TOKEN.findall(text))


class HFTokenCounter:
    """The embedding model's own tokenizer (`tokenizers`, tokenizer.json from the HF hub, no weights).

    Loaded on first use. If the tokenizer cannot be loaded (offline, hub down), it logs a warning
    and falls back to `RegexTokenCounter`, which undercounts Urdu slightly.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._tokenizer: Any = None
        self._fallback: RegexTokenCounter | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._tokenizer is not None or self._fallback is not None:
                return
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_pretrained(self.model_name)
            except Exception as exc:  # network, hub or file errors
                log.warning("tokenizer.load_failed", model=self.model_name, error=type(exc).__name__)
                self._fallback = RegexTokenCounter()

    def count(self, text: str) -> int:
        if self._tokenizer is None and self._fallback is None:
            self._load()
        if self._fallback is not None:
            return self._fallback.count(text)
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


# ---------------------------------------------------------------- chunks


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str  # clean display text
    char_start: int
    char_end: int
    token_count: int
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    is_table: bool = False
    is_ocr: bool = False
    language: str = "en"
    script: str = "Latn"


def context_header(
    agency: str | None, jurisdiction: str | None, title: str | None, section_path: Sequence[str]
) -> str:
    """`{agency} ({jurisdiction}) › {title} › {section}`, skipping the parts that are unknown."""
    parts: list[str] = []
    if agency and jurisdiction:
        parts.append(f"{agency} ({jurisdiction})")
    elif agency or jurisdiction:
        parts.append(agency or jurisdiction or "")
    if title:
        parts.append(title)
    if section_path:
        parts.append(" › ".join(section_path))
    return " › ".join(parts)


def embedding_text(header: str, text: str) -> str:
    return f"{header}\n\n{text}" if header else text


Span = tuple[int, int]


class _Splitter:
    def __init__(self, text: str, counter: TokenCounter, limit: int, overlap: int) -> None:
        self.text = text
        self.counter = counter
        self.limit = limit
        self.overlap = overlap

    def tokens(self, s: int, e: int) -> int:
        return self.counter.count(self.text[s:e])

    def fits(self, s: int, e: int) -> bool:
        return self.tokens(s, e) <= self.limit

    # ---- recursive split of one span into adjacent pieces that each fit
    def split(self, s: int, e: int, level: int = 0) -> list[Span]:
        """Cut a span into adjacent pieces that each fit.

        Down to sentence level a span is always cut, so `pack` gets fine pieces and has room to
        carry an overlap; below that (Urdu comma, commas, words, characters) only when it is needed.
        """
        for k in range(level, len(SEPARATORS)):
            if k >= _FINE_LEVELS and self.fits(s, e):
                return [(s, e)]
            sep = SEPARATORS[k]
            cuts: list[int] = []
            pos = self.text.find(sep, s, e)
            while pos != -1:
                cut = pos + len(sep)
                if s < cut < e:
                    cuts.append(cut)
                pos = self.text.find(sep, cut, e)
            if not cuts:
                continue
            out: list[Span] = []
            for ps, pe in itertools.pairwise([s, *cuts, e]):
                out.extend(self.split(ps, pe, k + 1))
            return out
        return [(s, e)] if self.fits(s, e) else self._hard_split(s, e)

    def _hard_split(self, s: int, e: int) -> list[Span]:
        out: list[Span] = []
        while s < e:
            lo, hi = s + 1, e  # largest m with text[s:m] fitting; at least one character
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.fits(s, mid):
                    lo = mid
                else:
                    hi = mid - 1
            out.append((s, lo))
            s = lo
        return out

    # ---- overlap: the earliest word boundary in (floor, end) whose tail fits in `overlap` tokens
    def overlap_start(self, floor: int, end: int) -> int | None:
        if self.overlap <= 0:
            return None
        window_start = max(floor + 1, end - self.overlap * 16)
        text = self.text
        bounds = [
            p for p in range(window_start, end) if not text[p].isspace() and (p == 0 or text[p - 1].isspace())
        ]
        if not bounds:
            return None
        lo, hi = 0, len(bounds)
        while lo < hi:  # token counts shrink as the start moves right
            mid = (lo + hi) // 2
            if self.tokens(bounds[mid], end) <= self.overlap:
                hi = mid
            else:
                lo = mid + 1
        return bounds[lo] if lo < len(bounds) else None

    # ---- pack pieces into chunks with overlap
    def pack(self, pieces: list[Span]) -> list[Span]:
        chunks: list[Span] = []
        i = 0
        start = pieces[0][0] if pieces else 0
        while i < len(pieces):
            if not self.fits(start, pieces[i][1]):
                start = pieces[i][0]  # overlap would overflow: start fresh
            end = pieces[i][1]
            j = i + 1
            while j < len(pieces) and self.fits(start, pieces[j][1]):
                end = pieces[j][1]
                j += 1
            chunks.append((start, end))
            i = j
            if i < len(pieces):
                ov = self.overlap_start(_strip_span(self.text, start, end)[0], end)
                start = ov if ov is not None else pieces[i][0]
        return chunks


def _strip_span(text: str, s: int, e: int) -> Span:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _table_parts(block: ExtractedBlock, splitter: _Splitter) -> list[tuple[str, int, int]]:
    """Split a Markdown table by rows, repeating the header (and its separator row) in each part."""
    text = block.text
    if splitter.counter.count(text) <= splitter.limit:
        return [(text, block.char_start, block.char_start + len(text))]
    lines = text.split("\n")
    offsets = []
    pos = block.char_start
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    n_header = 2 if len(lines) > 1 and _TABLE_SEPARATOR_ROW.match(lines[1].strip()) else 1
    header = "\n".join(lines[:n_header])
    if splitter.counter.count(header) > splitter.limit // 2:  # absurd header: plain split
        header, n_header = "", 0
    parts: list[tuple[str, int, int]] = []
    rows: list[int] = []

    def flush() -> None:
        if not rows:
            return
        body = "\n".join(lines[r] for r in rows)
        part = f"{header}\n{body}" if header else body
        start = block.char_start if not parts else offsets[rows[0]]
        parts.append((part, start, offsets[rows[-1]] + len(lines[rows[-1]])))
        rows.clear()

    for r in range(n_header, len(lines)):
        candidate = "\n".join([header, *(lines[x] for x in rows), lines[r]]) if header else lines[r]
        if rows and splitter.counter.count(candidate) > splitter.limit:
            flush()
        row_alone = f"{header}\n{lines[r]}" if header else lines[r]
        if splitter.counter.count(row_alone) > splitter.limit:
            flush()  # one giant row: split its text like prose (header not repeated)
            rs = offsets[r]
            for ps, pe in splitter.split(rs, rs + len(lines[r])):
                parts.append((splitter.text[ps:pe], ps, pe))
            continue
        rows.append(r)
    flush()
    return parts


def chunk_blocks(
    blocks: Sequence[ExtractedBlock],
    *,
    counter: TokenCounter,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Chunk cleaned blocks (from `cleaning.clean_blocks`, so `char_start` indexes the joined text)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))
    full = join_blocks(list(blocks))
    splitter = _Splitter(full, counter, chunk_size, overlap)
    raw: list[tuple[str, int, int, list[ExtractedBlock], bool]] = []

    def emit_run(run: list[ExtractedBlock]) -> None:
        if not run:
            return
        pieces: list[Span] = []
        for b in run:
            pieces.extend(splitter.split(b.char_start, b.char_start + len(b.text)))
        last_end = -1
        for s, e in splitter.pack(pieces):
            s, e = _strip_span(full, s, e)
            if s >= e or e <= last_end:  # nothing new beyond the overlap (whitespace only)
                continue
            last_end = e
            members = [b for b in run if b.char_start < e and b.char_start + len(b.text) > s]
            raw.append((full[s:e], s, e, members, False))

    # Group consecutive blocks by section; tables break text runs and stay atomic.
    run: list[ExtractedBlock] = []
    section: tuple[str, ...] | None = None
    for block in blocks:
        if block.section_path != section:
            emit_run(run)
            run = []
            section = block.section_path
        if block.kind == "table":
            emit_run(run)
            run = []
            for text, s, e in _table_parts(block, splitter):
                raw.append((text, s, e, [block], True))
        else:
            run.append(block)
    emit_run(run)

    chunks: list[Chunk] = []
    for index, (text, s, e, members, is_table) in enumerate(raw):
        pages = [b.page for b in members if b.page is not None]
        chunks.append(
            Chunk(
                index=index,
                text=text,
                char_start=s,
                char_end=e,
                token_count=counter.count(text),
                section_path=members[0].section_path if members else (),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                is_table=is_table,
                is_ocr=any(b.is_ocr for b in members),
                language=detect_language(text),
                script=detect_script(text),
            )
        )
    return chunks
