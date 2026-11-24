"""LexicalSparseEncoder: deterministic hashed bag-of-words sparse vectors for hosted embeddings."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from app.protocols import SparseVector
from app.retrieval.hosted import LexicalSparseEncoder


@pytest.fixture
def enc() -> LexicalSparseEncoder:
    return LexicalSparseEncoder()


def only_index(vec: SparseVector) -> int:
    assert len(vec.indices) == 1
    return vec.indices[0]


def test_returns_sparse_vector_with_valid_shape(enc: LexicalSparseEncoder) -> None:
    vec = enc.encode_document("passport renewal fee passport office")
    assert isinstance(vec, SparseVector)
    assert len(vec.indices) == len(vec.values) == 4
    assert list(vec.indices) == sorted(set(vec.indices))
    assert all(isinstance(i, int) and 0 <= i < 2**31 for i in vec.indices)
    assert all(v > 0 for v in vec.values)


def test_same_token_same_index_in_documents_and_queries(enc: LexicalSparseEncoder) -> None:
    doc = enc.encode_document("passport")
    query = enc.encode_query("passport")
    assert only_index(doc) == only_index(query)


def test_deterministic_across_processes(enc: LexicalSparseEncoder) -> None:
    code = (
        "from app.retrieval.hosted import LexicalSparseEncoder as E;"
        "v=E().encode_document('passport renewal fee');print(v.indices, v.values)"
    )
    outs = {
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3")
    }
    v = enc.encode_document("passport renewal fee")
    assert outs == {f"{v.indices} {v.values}\n"}


def test_case_insensitive(enc: LexicalSparseEncoder) -> None:
    assert enc.encode_document("Passport") == enc.encode_document("passport")
    assert enc.encode_query("PASSPORT") == enc.encode_query("passport")


@pytest.mark.parametrize(
    ("variant", "canonical"),
    [
        ("كتاب", "کتاب"),  # Arabic kaf -> keheh
        ("ياد", "یاد"),  # Arabic yeh -> Farsi yeh
        ("هوائی", "ہوائی"),  # Arabic heh -> heh goal
    ],
)
def test_urdu_letter_variants_unify(enc: LexicalSparseEncoder, variant: str, canonical: str) -> None:
    assert enc.encode_query(variant) == enc.encode_query(canonical)
    assert enc.encode_document(variant) == enc.encode_document(canonical)


def test_urdu_passport_with_arabic_variants_same_index(enc: LexicalSparseEncoder) -> None:
    canonical = "پاسپورٹ"  # passport, no variant letters
    assert only_index(enc.encode_query(canonical)) == only_index(enc.encode_document(canonical))
    with_yeh = "یاد"
    arabic_yeh = with_yeh.replace("ی", "ي")
    assert only_index(enc.encode_query(with_yeh)) == only_index(enc.encode_query(arabic_yeh))


@pytest.mark.parametrize(
    "text",
    ["the", "is of the", "ka ki hai", "کا ہے", "The IS Of"],
)
def test_stopword_only_text_is_empty(enc: LexicalSparseEncoder, text: str) -> None:
    for vec in (enc.encode_document(text), enc.encode_query(text)):
        assert vec.indices == ()
        assert vec.values == ()


def test_stopwords_dropped_but_content_kept(enc: LexicalSparseEncoder) -> None:
    noisy = enc.encode_query("the passport is of ka ki hai کا ہے")
    assert noisy == enc.encode_query("passport")


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "!!! ???"])
def test_empty_text_gives_empty_vector(enc: LexicalSparseEncoder, text: str) -> None:
    assert enc.encode_document(text).indices == ()
    assert enc.encode_query(text).indices == ()


def test_repeated_term_weighs_more_with_saturation(enc: LexicalSparseEncoder) -> None:
    single = enc.encode_document("passport")
    triple = enc.encode_document("passport passport passport")
    ten = enc.encode_document(" ".join(["passport"] * 10))
    assert only_index(single) == only_index(ten)
    assert single.values[0] < triple.values[0] < ten.values[0]
    assert ten.values[0] < 10 * single.values[0]


def test_query_has_one_entry_per_unique_token(enc: LexicalSparseEncoder) -> None:
    vec = enc.encode_query("passport passport fee the fee office")
    assert len(vec.indices) == 3
    assert list(vec.indices) == sorted(vec.indices)
    assert all(v > 0 for v in vec.values)


def test_roman_urdu_content_words_survive(enc: LexicalSparseEncoder) -> None:
    vec = enc.encode_query("passport ki fee kitni hai")
    assert len(vec.indices) == 3  # passport, fee, kitni
