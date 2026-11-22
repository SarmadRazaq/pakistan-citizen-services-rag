"""Structure-aware chunking (blueprint §3.3), including a hypothesis property test."""

from __future__ import annotations

import itertools

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.ingestion.chunking import (
    HFTokenCounter,
    RegexTokenCounter,
    chunk_blocks,
    context_header,
    embedding_text,
)
from app.ingestion.cleaning import clean_blocks, join_blocks
from app.protocols import ExtractedBlock

COUNTER = RegexTokenCounter()


def _blocks(*texts: str, section: tuple[str, ...] = ()) -> list[ExtractedBlock]:
    return clean_blocks(
        [ExtractedBlock(text=t, kind="paragraph", page=1, section_path=section) for t in texts]
    )


def test_small_document_is_one_chunk() -> None:
    blocks = _blocks("First paragraph.", "Second paragraph.")
    chunks = chunk_blocks(blocks, counter=COUNTER, chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."
    assert (chunks[0].char_start, chunks[0].char_end) == (0, len(join_blocks(blocks)))


def test_long_text_splits_on_sentences_with_overlap() -> None:
    sentences = [f"Sentence number {i} talks about the passport fee." for i in range(40)]
    blocks = _blocks(" ".join(sentences))
    chunks = chunk_blocks(blocks, counter=COUNTER, chunk_size=40, overlap=8)
    assert len(chunks) > 3
    assert all(c.token_count <= 40 for c in chunks)
    for prev, cur in itertools.pairwise(chunks):
        assert cur.char_start < prev.char_end  # overlap
        assert COUNTER.count(join_blocks(blocks)[cur.char_start : prev.char_end]) <= 8


def test_urdu_sentence_separators() -> None:
    urdu = "۔ ".join(["نادرا کے دفتر میں شناختی کارڈ کی درخواست جمع کرائی جاتی ہے"] * 12) + "۔"
    chunks = chunk_blocks(_blocks(urdu), counter=COUNTER, chunk_size=30, overlap=0)
    assert len(chunks) > 1
    assert all(c.text.endswith("۔") for c in chunks)  # split after the Urdu full stop
    assert all(c.language == "ur" and c.script == "Arab" for c in chunks)


def test_sections_are_hard_boundaries() -> None:
    blocks = clean_blocks(
        [
            ExtractedBlock(text="Fees", kind="heading", section_path=("Fees",)),
            ExtractedBlock(text="Normal fee text.", kind="paragraph", section_path=("Fees",)),
            ExtractedBlock(text="Documents", kind="heading", section_path=("Documents",)),
            ExtractedBlock(text="Bring your CNIC.", kind="paragraph", section_path=("Documents",)),
        ]
    )
    chunks = chunk_blocks(blocks, counter=COUNTER, chunk_size=200, overlap=10)
    assert [(c.text, c.section_path) for c in chunks] == [
        ("Fees\n\nNormal fee text.", ("Fees",)),
        ("Documents\n\nBring your CNIC.", ("Documents",)),
    ]


def test_tables_are_atomic_and_split_by_rows_with_header() -> None:
    header = "| Category | Normal | Urgent |\n|---|---|---|"
    rows = [f"| Category {i} | {i * 100} | {i * 200} |" for i in range(30)]
    table = ExtractedBlock(text=header + "\n" + "\n".join(rows), kind="table", page=3)
    blocks = clean_blocks([ExtractedBlock(text="Intro text.", kind="paragraph", page=3), table])
    small = chunk_blocks(blocks, counter=COUNTER, chunk_size=10_000, overlap=0)
    assert [c.is_table for c in small] == [False, True]  # the table never merges with prose
    parts = chunk_blocks(blocks, counter=COUNTER, chunk_size=80, overlap=10)
    tables = [c for c in parts if c.is_table]
    assert len(tables) > 1
    for part in tables:
        assert part.text.startswith(header + "\n")
        assert part.token_count <= 80
        assert part.page_start == part.page_end == 3
    body_rows = [line for part in tables for line in part.text.split("\n")[2:]]
    assert body_rows == rows  # every row exactly once, in order


def test_context_header() -> None:
    assert context_header("DGIP", "federal", "Passport Fees", ("Urgent",)) == (
        "DGIP (federal) › Passport Fees › Urgent"
    )
    assert context_header(None, None, "Notes", ()) == "Notes"
    assert embedding_text("", "text") == "text"
    assert embedding_text("H", "text") == "H\n\ntext"


def test_hf_counter_falls_back_without_network(monkeypatch: object) -> None:
    import tokenizers

    def boom(*args: object, **kwargs: object) -> object:
        raise OSError("offline")

    monkeypatch.setattr(tokenizers.Tokenizer, "from_pretrained", boom)  # type: ignore[attr-defined]
    counter = HFTokenCounter("BAAI/bge-m3")
    assert counter.count("two words") == 2


# ---------------------------------------------------------------- property test

_alphabet = st.sampled_from([*"abcdefghij ABC.,!?۔؟،\n", "نادرا", "کے", "fee", "  ", "\n\n", "x" * 30])
_paragraph = st.lists(_alphabet, min_size=1, max_size=60).map("".join)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    paragraphs=st.lists(_paragraph, min_size=1, max_size=8),
    chunk_size=st.integers(min_value=3, max_value=40),
    overlap=st.integers(min_value=0, max_value=12),
)
def test_chunking_properties(paragraphs: list[str], chunk_size: int, overlap: int) -> None:
    blocks = _blocks(*paragraphs)
    full = join_blocks(blocks)
    chunks = chunk_blocks(blocks, counter=COUNTER, chunk_size=chunk_size, overlap=overlap)

    covered = [False] * len(full)
    for chunk in chunks:
        assert chunk.text == full[chunk.char_start : chunk.char_end]  # exact slice of the document
        assert chunk.text.strip() == chunk.text and chunk.text
        assert COUNTER.count(chunk.text) <= chunk_size  # size ≤ limit
        for i in range(chunk.char_start, chunk.char_end):
            covered[i] = True
    # no text lost: every non-whitespace character is inside some chunk
    assert all(covered[i] for i, ch in enumerate(full) if not ch.isspace())
    # positions are monotonic
    starts = [c.char_start for c in chunks]
    ends = [c.char_end for c in chunks]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert ends == sorted(ends) and len(set(ends)) == len(ends)
    assert [c.index for c in chunks] == list(range(len(chunks)))
