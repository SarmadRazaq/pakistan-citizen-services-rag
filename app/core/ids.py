"""Identifiers and content hashes (blueprint §5.2). Shared by ingestion, auth and retrieval."""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from pathlib import Path

# Fixed namespace so point ids are stable across machines and re-runs.
POINT_NAMESPACE = uuid.UUID("5b1c1f6e-6a35-4f0b-9a52-3c7f1d2e9a10")


def new_user_id() -> str:
    return f"u_{uuid.uuid4().hex}"


def new_document_id() -> str:
    return f"d_{uuid.uuid4().hex}"


def new_feedback_id() -> str:
    return f"f_{uuid.uuid4().hex}"


def normalize_for_hash(text: str) -> str:
    """NFKC + collapsed whitespace, so cosmetic differences do not change content hashes."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def point_id(doc_id: str, chunk_hash: str, occurrence: int = 0) -> str:
    """Deterministic Qdrant point id: the same chunk of the same document always maps to the same point."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{doc_id}:{chunk_hash}:{occurrence}"))
