"""`python -m evals.run` end to end against a fake service (ASGITransport) and a fake NVIDIA API
(MockTransport), then an `--offline` re-run from the caches with no transport at all."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from evals import calibration
from evals.llm import RateLimiter
from evals.run import NoItemsError, RunOptions, main, parse_args, run_eval
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings

REPO = Path(__file__).resolve().parents[2]
POLICY = "ID-05__nadra-registration-policy-6-0-2-en.pdf"
JUDGE_MODEL = "openai/gpt-oss-120b"


def _item(id_: str, question: str, **over: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": id_,
        "question": question,
        "language": "en",
        "answer_language": "en",
        "type": "single_fact",
        "answerable": True,
        "gold_sources": [{"source_id": "ID-05", "pages": [13]}],
        "agencies": ["NADRA"],
        "jurisdictions": ["federal"],
        "expected_notices": [],
        "source_ids": ["ID-05"],
        "reference_answer": "A verified computerized birth certificate.",
        "split": "test",
        "review_status": "approved",
        "author": "student",
        "reviewed_by": "student",
    }
    data.update(over)
    return data


ITEMS = [
    _item("q_t_en", "What do I need to change my name on my CNIC?"),
    _item(
        "q_t_ru",
        "Shadi ke baad marital status kaise change karwaun?",
        language="roman_ur",
        answer_language="roman_ur",
        gold_sources=[{"source_id": "ID-05", "pages": [15]}],
        reference_answer="A marriage certificate from the Union Council.",
    ),
    _item(
        "q_t_adv",
        "Mera court case chal raha hai, date of birth ka kya karun?",
        language="roman_ur",
        answer_language="roman_ur",
        type="advice_boundary",
        expected_notices=["not_advice", "referral"],
        gold_sources=[{"source_id": "ID-05", "pages": [14]}],
        reference_answer="General rule only, plus the not-advice notice and referral.",
    ),
    _item(
        "q_t_un",
        "How many days does NADRA take to process a date-of-birth change?",
        type="unanswerable_near_miss",
        answerable=False,
        gold_sources=[],
        reference_answer="Refuse: the policy gives no processing time.",
    ),
    _item(
        "q_t_ur",
        "شناختی کارڈ پر تاریخ پیدائش بدلنی ہو تو کیا چاہیے؟",
        language="ur",
        answer_language="ur",
        gold_sources=[{"source_id": "ID-05", "pages": [14]}],
        reference_answer="Birth certificate and DG approval for big changes.",
    ),
    _item("q_t_draft", "A draft question that must be skipped?", review_status="draft", reviewed_by=None),
]


def _chunk(rank: int, file_name: str, page: int, used: bool = False) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": f"{file_name}:{page}:{rank}",
        "document_id": "d_1",
        "scope": "public",
        "file_name": file_name,
        "page": page,
        "chunk_index": rank,
        "text": f"text of page {page}",
        "similarity": 0.9 - rank / 100,
        "used_in_context": used,
    }


# question id -> (search chunks, ask body overrides)
def _canned(qid: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if qid == "q_t_en":
        chunks = [_chunk(1, POLICY, 2, True), _chunk(2, POLICY, 13, True), _chunk(3, "other.pdf", 1)]
        return chunks, {
            "answer": "You need a verified computerized birth certificate [S1].",
            "citations": [_cite(POLICY, 13, chunks[1]["chunk_id"])],
        }
    if qid == "q_t_ru":
        chunks = [_chunk(1, POLICY, 15, True)]
        return chunks, {
            "answer": "Aap ko Union Council se marriage certificate chahiye [S1].",
            "citations": [_cite(POLICY, 15, chunks[0]["chunk_id"])],
        }
    if qid == "q_t_adv":
        chunks = [_chunk(1, POLICY, 14, True)]
        return chunks, {
            "answer": "Aam qaida yeh hai ke birth certificate chahiye [S1]. Yeh qanooni mashwara nahi hai.",
            "citations": [_cite(POLICY, 14, chunks[0]["chunk_id"])],
            "notices": [{"code": "not_advice", "message": "General information, not legal advice."}],
        }
    if qid == "q_t_un":
        return [_chunk(1, POLICY, 14)], {
            "answer": "The documents do not say how long this takes.",
            "answerable": False,
            "refusal_reason": "insufficient_context",
        }
    # q_t_ur: retrieval miss, and the answer comes back in English
    return [_chunk(1, "other.pdf", 3, True)], {
        "answer": "You need a birth certificate [S1].",
        "citations": [_cite("other.pdf", 3, "other.pdf:3:1")],
    }


def _cite(file_name: str, page: int, chunk_id: str) -> dict[str, Any]:
    return {
        "marker": "S1",
        "chunk_id": chunk_id,
        "document_id": "d_1",
        "scope": "public",
        "file_name": file_name,
        "page": page,
        "quote": "quoted text",
    }


QUESTION_TO_ID = {i["question"]: i["id"] for i in ITEMS}


def fake_service(*, honour_header: bool, calls: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()

    def _authorized(request: Request) -> bool:
        return request.headers.get("authorization") == "Bearer tok-1"

    @app.post("/api/v1/auth/login")
    async def login(request: Request) -> JSONResponse:
        form = await request.form()
        calls.append({"path": "login"})
        if form.get("username") != "eval@example.com" or form.get("password") != "pw-123456789":
            return JSONResponse({"code": "invalid_credentials"}, status_code=401)
        return JSONResponse({"access_token": "tok-1", "token_type": "bearer", "expires_in": 1800})

    def _headers(request: Request) -> dict[str, str]:
        header = request.headers.get("x-eval-config")
        if honour_header and header:
            return {"X-Eval-Config-Applied": ",".join(sorted(k for k in json.loads(header) if k != "config"))}
        return {}

    @app.post("/api/v1/search")
    async def search(request: Request) -> JSONResponse:
        if not _authorized(request):
            return JSONResponse({"code": "unauthorized"}, status_code=401)
        body = await request.json()
        calls.append({"path": "search", "body": body, "header": request.headers.get("x-eval-config")})
        chunks, _ = _canned(QUESTION_TO_ID[body["query"]])
        retrieval = {
            "query_variants": [body["query"]],
            "candidates_fetched": len(chunks),
            "chunks_used": 0,
            "threshold": 0.3,
            "chunks": chunks,
        }
        return JSONResponse(
            {"request_id": "r", "retrieval": retrieval, "latency_ms": {"total": 50}},
            headers=_headers(request),
        )

    @app.post("/api/v1/ask")
    async def ask(request: Request) -> JSONResponse:
        if not _authorized(request):
            return JSONResponse({"code": "unauthorized"}, status_code=401)
        body = await request.json()
        calls.append({"path": "ask", "body": body, "header": request.headers.get("x-eval-config")})
        chunks, over = _canned(QUESTION_TO_ID[body["question"]])
        resp: dict[str, Any] = {
            "request_id": "r",
            "answer": "",
            "answerable": True,
            "refusal_reason": None,
            "notices": [],
            "language": {"question": "en", "answer": "en"},
            "citations": [],
            "retrieval": {
                "query_variants": [],
                "candidates_fetched": len(chunks),
                "chunks_used": 1,
                "threshold": 0.3,
                "chunks": chunks,
            },
            "provider": {"endpoint": "nvidia", "model": "deepseek-ai/deepseek-v3.1", "attempts": 1},
            "usage": {"input_tokens": 3000, "output_tokens": 120},
            "latency_ms": {"total": 1000 if body["question"].isascii() else 3000},
        }
        resp.update(over)
        return JSONResponse(resp, headers=_headers(request))

    return app


class FakeNvidia:
    """Judge verdicts and closed-book answers, keyed on what the prompt contains."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        self.calls.append(model)
        prompt = body["messages"][-1]["content"]
        if model == JUDGE_MODEL:
            closed = "CLOSED-BOOK" in prompt
            wrong = "شناختی" in prompt or (closed and "marital" in prompt)
            verdict = {
                "correctness": 0 if wrong else (1 if closed else 2),
                "citation_support": "na" if closed else "yes",
                "refusal_appropriate": True,
                "disclaimer_present": "qanooni mashwara nahi" in prompt,
                "personal_directive": False,
                "rationale": "wrong facts" if wrong else "matches the reference",
            }
            content = json.dumps(verdict)
        else:
            content = "From memory: a birth certificate is usually needed."
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 9},
            },
        )


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in ITEMS) + "\n", encoding="utf-8")
    return path


def _opts(tmp_path: Path, golden: Path, **over: Any) -> RunOptions:
    eval_md = tmp_path / "EVAL.md"
    if not eval_md.exists():
        eval_md.write_text((REPO / "EVAL.md").read_text(encoding="utf-8"), encoding="utf-8")
    base: dict[str, Any] = {
        "base_url": "http://svc.test",
        "email": "eval@example.com",
        "password": "pw-123456789",
        "split": "test",
        "golden_path": golden,
        "results_dir": tmp_path / "results",
        "cache_dir": tmp_path / "cache",
        "eval_md": eval_md,
        "run_name": "live",
        "concurrency": 2,
    }
    base.update(over)
    return RunOptions(**base)


def _settings(**over: Any) -> Settings:
    return Settings(_env_file=None, **over)  # type: ignore[call-arg]


def _fast() -> RateLimiter:
    return RateLimiter(10**9)


def _rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["config"], r["id"]): r for r in csv.DictReader(fh)}


async def test_end_to_end_then_offline_rerun(tmp_path: Path, golden: Path) -> None:
    calls: list[dict[str, Any]] = []
    nvidia = FakeNvidia()
    out = await run_eval(
        _opts(tmp_path, golden),
        settings=_settings(nvidia_api_key="nvapi-test"),
        service_transport=httpx.ASGITransport(app=fake_service(honour_header=True, calls=calls)),
        llm_transport=httpx.MockTransport(nvidia),
        limiter=_fast(),
    )
    for name in ("metrics.json", "per_question.csv", "answers.jsonl", "report.md"):
        assert (out / name).exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    configs = metrics["configs"]
    assert set(configs) == {"CB_closed_book", "R1_dense_only", "R4_hybrid_rerank"}
    assert metrics["run"]["golden"]["evaluated"] == 5  # the draft item was skipped

    # one login, then /search + /ask per item per service config
    assert sum(c["path"] == "login" for c in calls) == 1
    asks = [c for c in calls if c["path"] == "ask"]
    assert len(asks) == 10
    assert all(a["body"]["scope"] == "public" and a["body"]["answer_language"] == "auto" for a in asks)
    r1_headers = {a["header"] for a in asks if a["header"]}
    assert r1_headers and json.loads(r1_headers.pop())["retrieval_mode"] == "dense"
    assert configs["R1_dense_only"]["meta"]["override_honoured"] == "yes"
    assert configs["R4_hybrid_rerank"]["meta"]["override_honoured"] == "n/a"

    base = configs["R4_hybrid_rerank"]
    overall = base["overall"]
    assert overall["n"] == 5 and overall["n_answerable"] == 4
    assert overall["hit@1"] == pytest.approx(0.5)  # en hits at rank 2, ru + adv at 1, ur misses
    assert overall["hit@5"] == pytest.approx(0.75)
    assert overall["mrr@10"] == pytest.approx((0.5 + 1 + 1 + 0) / 4)
    assert overall["refusal_recall"] == 1.0 and overall["refusal_precision"] == 1.0
    assert overall["false_refusal_rate"] == 0.0
    assert overall["citation_validity"] == pytest.approx(0.75)  # the ur answer cites the wrong file
    assert overall["notices_ok"] == 0.0  # the advice item lacks the referral notice
    assert overall["latency_p50_ms"] == 1000.0
    assert base["by_language"]["ur"]["language_match"] == 0.0  # English answer to an Urdu question
    assert base["by_language"]["roman_ur"]["language_match"] == 1.0
    assert base["by_language"]["en"]["hit@5"] == 1.0
    assert base["failures"] == {"advice_boundary": 1, "ok": 3, "retrieval_miss": 1}

    cb = configs["CB_closed_book"]
    assert cb["meta"]["kind"] == "closed_book" and cb["meta"]["model"] == "deepseek-ai/deepseek-v3.1"
    assert cb["overall"]["n"] == 4  # answerable items only
    assert cb["overall"]["judge_correctness_mean"] == pytest.approx((1 + 0 + 1 + 0) / 4)

    rows = _rows(out / "per_question.csv")
    en = rows[("R4_hybrid_rerank", "q_t_en")]
    assert en["mrr@10"] == "0.5" and en["judge_correctness"] == "2" and en["failure_category"] == "ok"
    adv = rows[("R4_hybrid_rerank", "q_t_adv")]
    assert (
        adv["judge_advice_ok"] == "1"
        and adv["notices_ok"] == "0"
        and adv["failure_category"] == "advice_boundary"
    )
    assert rows[("R4_hybrid_rerank", "q_t_un")]["refused"] == "1"

    report_md = (out / "report.md").read_text(encoding="utf-8")
    assert "**R4_hybrid_rerank** (baseline)" in report_md and "closed-book (no sources)" in report_md
    eval_md = (tmp_path / "EVAL.md").read_text(encoding="utf-8")
    assert "R1_dense_only" in eval_md and "_Not run yet._" not in eval_md

    judge_calls = sum(m == JUDGE_MODEL for m in nvidia.calls)
    # R4 (5) + CB (4); R1's answers equal R4's here, so its identical judge prompts hit the disk cache
    assert judge_calls == 4 + 5

    # ---- offline: no service, no NVIDIA transport, no key; identical numbers from the caches
    offline = await run_eval(
        _opts(tmp_path, golden, offline=True, run_name="offline", email=None, password=None),
        settings=_settings(),
        limiter=_fast(),
    )
    again = json.loads((offline / "metrics.json").read_text(encoding="utf-8"))
    assert again["run"]["network_calls"] == {"service": 0, "judge": 0, "closed_book": 0}
    for name, data in configs.items():
        data["meta"].pop("seconds")
        again["configs"][name]["meta"].pop("seconds")
        assert again["configs"][name] == data
    assert _rows(offline / "per_question.csv") == rows

    # ---- calibration export from the run, blind to the judge
    labels = tmp_path / "human_labels.csv"
    n = calibration.export_labels(out, labels, n=30)
    assert n == 5
    with labels.open(newline="", encoding="utf-8-sig") as fh:
        exported = list(csv.DictReader(fh))
    assert "judge_correctness" not in exported[0] and exported[0]["human_correctness"] == ""
    for row in exported:
        row["human_correctness"] = "2" if row["id"] != "q_t_ur" else "0"
        row["human_citation_supported"] = "y"
    with labels.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    cal = calibration.score_labels(out, labels)
    assert cal["correctness"]["n"] == 5 and cal["correctness"]["kappa"] == pytest.approx(1.0)
    with pytest.raises(FileExistsError):
        calibration.export_labels(out, labels)


async def test_unhonoured_overrides_fall_back_and_errors_are_reported(tmp_path: Path, golden: Path) -> None:
    calls: list[dict[str, Any]] = []
    out = await run_eval(
        _opts(tmp_path, golden, configs=["R1_dense_only"], judge=False, eval_md=None, password="wrong-pass"),
        settings=_settings(),
        service_transport=httpx.ASGITransport(app=fake_service(honour_header=False, calls=calls)),
        limiter=_fast(),
    )
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    r1 = metrics["configs"]["R1_dense_only"]
    assert r1["overall"]["errors"] == 5 and r1["failures"] == {"error": 5}
    assert r1["meta"]["override_honoured"] == "unknown"
    assert sum(c["path"] == "login" for c in calls) == 1  # a failed login is not retried per item

    out2 = await run_eval(
        _opts(tmp_path, golden, configs=["R1_dense_only"], judge=False, eval_md=None, run_name="r2"),
        settings=_settings(),
        service_transport=httpx.ASGITransport(app=fake_service(honour_header=False, calls=calls)),
        limiter=_fast(),
    )
    r1 = json.loads((out2 / "metrics.json").read_text(encoding="utf-8"))["configs"]["R1_dense_only"]
    assert r1["meta"]["override_honoured"] == "no"
    assert "NOT honoured" in (out2 / "report.md").read_text(encoding="utf-8")


async def test_no_cache_forces_fresh_calls(tmp_path: Path, golden: Path) -> None:
    calls: list[dict[str, Any]] = []
    app = fake_service(honour_header=False, calls=calls)
    for name, use_cache in (("a", True), ("b", True), ("c", False)):
        await run_eval(
            _opts(
                tmp_path,
                golden,
                configs=["R4_hybrid_rerank"],
                judge=False,
                eval_md=None,
                run_name=name,
                use_cache=use_cache,
                limit=2,
            ),
            settings=_settings(),
            service_transport=httpx.ASGITransport(app=app),
            limiter=_fast(),
        )
    asks = [c for c in calls if c["path"] == "ask"]
    assert len(asks) == 4  # run a: 2 fresh, run b: cached, run c (--no-cache): 2 fresh


async def test_no_items_is_a_clear_error(tmp_path: Path, golden: Path) -> None:
    with pytest.raises(NoItemsError, match="include-drafts"):
        await run_eval(_opts(tmp_path, golden, split="dev"), settings=_settings())


def test_cli_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_EMAIL", "e@example.com")
    monkeypatch.setenv("EVAL_PASSWORD", "secret-pass")
    opts = parse_args(
        [
            "--configs",
            "R4_hybrid_rerank",
            "CB_closed_book",
            "--split",
            "all",
            "--include-drafts",
            "--no-cache",
            "--limit",
            "3",
            "--no-eval-md",
        ]
    )
    assert opts.configs == ["R4_hybrid_rerank", "CB_closed_book"]
    assert opts.split == "all" and opts.include_drafts and not opts.use_cache and opts.limit == 3
    assert opts.email == "e@example.com" and opts.password == "secret-pass"
    assert opts.eval_md is None and opts.base_url == "http://127.0.0.1:8000"
    with pytest.raises(SystemExit):
        parse_args(["--offline", "--no-cache"])


def test_main_returns_2_without_items(tmp_path: Path, golden: Path) -> None:
    code = main(
        ["--golden", str(golden), "--split", "dev", "--results-dir", str(tmp_path / "r"), "--no-eval-md"]
    )
    assert code == 2
