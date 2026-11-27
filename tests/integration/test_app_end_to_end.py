"""End-to-end through the fully wired app (create_app → lifespan → real services).

Real: DB, auth (argon2/JWT), rate limits, ingestion (validation, extraction, chunking, dedupe),
HybridRetriever, LangGraph ask pipeline with quote verification, delete guarantees.
Faked: model weights (FakeEmbedder/FakeReranker), Qdrant (InMemoryVectorStore), and the LLM
(a scripted client that answers from the <source> blocks it is given). No network.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import pytest

from app.config import Settings
from app.core.health import HealthService
from app.protocols import LLMMessage, LLMResult
from app.retrieval.retriever import HybridRetriever
from tests.conftest import ok_probe
from tests.fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore

_SOURCE = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.S)


class SourceQuotingLLM:
    """Helper calls get a minimal understanding JSON; generate calls quote the first source verbatim."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        purpose: Literal["generate", "helper", "judge"],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResult:
        self.calls.append(purpose)
        if purpose == "helper":
            text = json.dumps({"english_query": "", "urdu_query": "", "sub_queries": [], "off_topic": False})
        else:
            joined = "\n".join(m.content for m in messages)
            sources = _SOURCE.findall(joined)
            if not sources:
                text = json.dumps({"answerable": False, "answer": "", "citations": []})
            else:
                sid, body = sources[0]
                quote = " ".join(body.split()[:8])
                text = json.dumps(
                    {
                        "answerable": True,
                        "answer": f"The urgent passport fee is listed in the schedule [{sid}].",
                        "citations": [{"source": sid, "quote": quote}],
                    }
                )
        return LLMResult(text=text, endpoint="hosted", model="fake", input_tokens=50, output_tokens=20)

    async def available_models(self) -> list[str]:
        return ["fake"]

    async def aclose(self) -> None:
        return None


@dataclass
class FakeBundle:
    store: InMemoryVectorStore
    embedder: FakeEmbedder
    retriever: HybridRetriever

    async def startup(self, *, warmup: bool = False) -> None:
        return None

    async def shutdown(self) -> None:
        return None


@pytest.fixture
async def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # tokenizer falls back to the regex counter
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'e2e.sqlite3').as_posix()}",
        storage_dir=tmp_path / "storage",
        jwt_secret="e2e-test-secret-value-that-is-long-enough",
        tau_refuse=0.05,
        tau_keep=0.0,
        ocr_enabled=False,
    )
    store, embedder = InMemoryVectorStore(), FakeEmbedder()
    bundle = FakeBundle(store, embedder, HybridRetriever(settings, embedder, store, FakeReranker()))
    llm = SourceQuotingLLM()
    monkeypatch.setattr("app.retrieval.factory.build_retrieval", lambda s: bundle)
    monkeypatch.setattr("app.generation.llm.build_llm_clients", lambda s, **kw: (llm, llm))

    from app.main import create_app

    app = create_app(
        settings,
        health_service=HealthService([ok_probe("qdrant")], timeout_s=1.0),
        wire_services=True,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app = app  # type: ignore[attr-defined]
            yield c


async def _token(c: httpx.AsyncClient, email: str) -> dict[str, str]:
    r = await c.post("/api/v1/auth/signup", json={"email": email, "password": "correct-horse-battery"})
    assert r.status_code == 201, r.text
    r = await c.post("/api/v1/auth/login", data={"username": email, "password": "correct-horse-battery"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


DOC = (
    "# Passport Fee Schedule\n\n"
    "Machine readable passport fees for 2025.\n\n"
    "Urgent processing of a 36 page passport costs Rs 9000 and takes five working days.\n\n"
    "Normal processing of a 36 page passport costs Rs 4500 and takes twenty one working days.\n"
)


async def test_signup_upload_ask_delete_flow(client: httpx.AsyncClient) -> None:
    a = await _token(client, "a@example.com")
    b = await _token(client, "b@example.com")

    # upload a private document as A
    files = {"file": ("Passport-Fee-Schedule.md", DOC.encode(), "text/markdown")}
    r = await client.post("/api/v1/documents", headers=a, files=files, data={"agency": "DGIP"})
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["status"] == "READY" and doc["chunk_count"] >= 1

    # same file again → 409 with the existing id
    r = await client.post("/api/v1/documents", headers=a, files=files)
    assert r.status_code == 409
    assert r.json()["extra"]["existing_document_id"] == doc["id"]

    # A asks: cited, verified answer, with every retrieved chunk and its scores
    r = await client.post("/api/v1/ask", headers=a, json={"question": "What is the urgent passport fee?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answerable"] is True
    assert body["citations"] and body["citations"][0]["document_id"] == doc["id"]
    chunks = body["retrieval"]["chunks"]
    assert chunks and all("similarity" in ch for ch in chunks)
    codes = {n["code"] for n in body["notices"]}
    assert "third_party_processing" in codes  # private upload sent to the hosted model (§3.7)

    # B cannot see A's document: list is empty, direct get is 404, B's answer has no corpus
    r = await client.get("/api/v1/documents", headers=b)
    assert r.status_code == 200 and r.json()["items"] == []
    assert (await client.get(f"/api/v1/documents/{doc['id']}", headers=b)).status_code == 404
    r = await client.post("/api/v1/ask", headers=b, json={"question": "What is the urgent passport fee?"})
    assert r.status_code == 200 and r.json()["answerable"] is False

    # delete → 204, vectors gone, the content can no longer be retrieved
    assert (await client.delete(f"/api/v1/documents/{doc['id']}", headers=a)).status_code == 204
    r = await client.post("/api/v1/search", headers=a, json={"query": "urgent passport fee"})
    assert r.status_code == 200 and r.json()["retrieval"]["chunks"] == []


async def test_unsupported_type_and_no_auth(client: httpx.AsyncClient) -> None:
    a = await _token(client, "c@example.com")
    r = await client.post(
        "/api/v1/documents",
        headers=a,
        files={"file": ("tool.pdf", b"MZ\x90\x00" + b"\x00" * 64, "application/pdf")},
    )
    assert r.status_code == 415
    assert (await client.get("/api/v1/documents")).status_code == 401
