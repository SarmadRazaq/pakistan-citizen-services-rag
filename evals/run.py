"""Run the evaluation:  python -m evals.run  (blueprint §10.7, MVP slice 5).

For every config and every golden item: /search and /ask on the running service (or, for the
closed-book config, the generator directly with no sources), then deterministic metrics, the LLM
judge, and aggregation overall / per type / per language / per split.

Outputs go to evals/results/<timestamp>/:
    metrics.json      every aggregate, plus run metadata (golden sha256, models, configs)
    per_question.csv  one row per (config, item): metric values, judge score, failure category
    answers.jsonl     the raw answers and citations (input for calibration.py)
    report.md         comparison tables across configs
and the generated tables in EVAL.md are refreshed.

Every service response and LLM call is cached on disk (evals/.cache/{api,judge,closed_book}/), so
`--offline` re-runs everything from the caches without the service or the NVIDIA API.

Examples
    python -m evals.run --include-drafts --split all              # today: the 10 draft items
    python -m evals.run --configs R4_hybrid_rerank CB_closed_book --split test
    python -m evals.run --offline --include-drafts --split all    # re-score from the caches
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from app.config import Settings
from evals import report
from evals.client import ApiResult, ServiceClient
from evals.eval_config import EvalConfig, resolve_configs
from evals.golden import GOLDEN_PATH, GoldenItem, golden_sha256, load_golden, read_items
from evals.judge import PROMPT_VERSION, Judge, JudgeInput, Verdict
from evals.llm import ChatClient, LLMError, RateLimiter
from evals.metrics import (
    ItemScore,
    breakdown,
    failure_category,
    is_refusal,
    notice_codes,
    ranked_chunks,
    score_closed_book_item,
    score_service_item,
)

EVALS = Path(__file__).resolve().parent
LANGUAGE_NAMES = {"en": "English", "ur": "Urdu (Arabic script)", "roman_ur": "Roman Urdu (Latin script)"}


class NoItemsError(RuntimeError):
    pass


@dataclass
class RunOptions:
    base_url: str = "http://127.0.0.1:8000"
    api_prefix: str = "/api/v1"
    email: str | None = None
    password: str | None = None
    configs: list[str] | None = None
    include_drafts: bool = False
    split: Literal["dev", "test", "all"] = "test"
    use_cache: bool = True
    offline: bool = False
    limit: int | None = None
    judge: bool = True
    concurrency: int = 2
    golden_path: Path = GOLDEN_PATH
    config_dir: Path = EVALS / "configs"
    results_dir: Path = EVALS / "results"
    cache_dir: Path = EVALS / ".cache"
    eval_md: Path | None = report.EVAL_MD
    run_name: str | None = None
    ragas: bool = False
    command: str = ""


@dataclass
class ItemRun:
    score: ItemScore
    answer: dict[str, Any] = field(default_factory=dict)
    applied: str | None = None
    responded: bool = False


def parse_args(argv: Sequence[str] | None = None) -> RunOptions:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--api-prefix", default="/api/v1")
    p.add_argument("--email", default=None, help="eval account (or env EVAL_EMAIL)")
    p.add_argument("--password", default=None, help="eval account password (or env EVAL_PASSWORD)")
    p.add_argument(
        "--configs", nargs="+", default=None, help="names, paths or globs; default: every default config"
    )
    p.add_argument("--include-drafts", action="store_true", help="also run draft items (not only approved)")
    p.add_argument("--split", choices=["dev", "test", "all"], default="test")
    p.add_argument(
        "--no-cache", action="store_true", help="ignore cached responses (fresh calls, cache rewritten)"
    )
    p.add_argument("--offline", action="store_true", help="use only the on-disk caches; no network at all")
    p.add_argument("--limit", type=int, default=None, help="first N items only (smoke runs)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    p.add_argument("--config-dir", type=Path, default=EVALS / "configs")
    p.add_argument("--results-dir", type=Path, default=EVALS / "results")
    p.add_argument("--cache-dir", type=Path, default=EVALS / ".cache")
    p.add_argument("--eval-md", type=Path, default=report.EVAL_MD)
    p.add_argument("--no-eval-md", action="store_true", help="do not touch EVAL.md")
    p.add_argument("--run-name", default=None, help="results folder name (default: UTC timestamp)")
    p.add_argument(
        "--ragas", action="store_true", help="also run RAGAS on the baseline (needs requirements-eval)"
    )
    a = p.parse_args(argv)
    if a.offline and a.no_cache:
        p.error("--offline reads only the caches, so it cannot be combined with --no-cache")
    return RunOptions(
        base_url=a.base_url,
        api_prefix=a.api_prefix,
        email=a.email or os.environ.get("EVAL_EMAIL"),
        password=a.password or os.environ.get("EVAL_PASSWORD"),
        configs=a.configs,
        include_drafts=a.include_drafts,
        split=a.split,
        use_cache=not a.no_cache,
        offline=a.offline,
        limit=a.limit,
        judge=not a.no_judge,
        concurrency=max(1, a.concurrency),
        golden_path=a.golden,
        config_dir=a.config_dir,
        results_dir=a.results_dir,
        cache_dir=a.cache_dir,
        eval_md=None if a.no_eval_md else a.eval_md,
        run_name=a.run_name,
        ragas=a.ragas,
        command="python -m evals.run " + " ".join(argv if argv is not None else sys.argv[1:]),
    )


# ---------------------------------------------------------------- one item


def _ask_language(cfg: EvalConfig, item: GoldenItem) -> str:
    lang = cfg.request.answer_language
    return item.answer_language if lang == "item" else lang


async def _service_item(cfg: EvalConfig, item: GoldenItem, service: ServiceClient) -> ItemRun:
    header = cfg.header_value()
    req = cfg.request
    search_body = {"query": item.question, "top_k": req.search_top_k, "scope": req.scope}
    ask_body = {
        "question": item.question,
        "top_k": req.top_k,
        "scope": req.scope,
        "answer_language": _ask_language(cfg, item),
    }
    search: ApiResult = await service.call("search", search_body, eval_header=header)
    ask: ApiResult = await service.call("ask", ask_body, eval_header=header)
    error = "; ".join(f"{r.endpoint}: {r.error}" for r in (search, ask) if r.error) or None
    score = score_service_item(
        cfg.name,
        item,
        search.body,
        ask.body,
        requested_language=_ask_language(cfg, item),
        client_latency_ms=ask.latency_ms,
        error=error,
    )
    answer: dict[str, Any] = {"config": cfg.name, "id": item.id, "question": item.question}
    if ask.body is not None:
        answer.update(
            answer=ask.body.get("answer", ""),
            refused=is_refusal(ask.body),
            refusal_reason=ask.body.get("refusal_reason"),
            notices=notice_codes(ask.body),
            citations=ask.body.get("citations") or [],
            contexts=[str(c.get("text") or "") for c in ranked_chunks(ask.body) if c.get("used_in_context")],
        )
    top = ranked_chunks(search.body)[:10]
    answer["retrieved_top10"] = [
        {"file_name": c.get("file_name"), "page": c.get("page"), "rank": c.get("rank")} for c in top
    ]
    return ItemRun(
        score=score,
        answer=answer,
        applied=search.applied or ask.applied,
        responded=search.body is not None or ask.body is not None,
    )


def closed_book_messages(item: GoldenItem) -> list[dict[str, str]]:
    lang = LANGUAGE_NAMES[item.answer_language]
    system = (
        "You answer questions about Pakistani citizen services (NADRA, passports, taxes, consumer rights, "
        "social protection) from your own knowledge only: no documents are provided. "
        f"Answer in {lang}, concisely (at most 120 words). If you do not know, say so."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": item.question}]


async def _closed_book_item(cfg: EvalConfig, item: GoldenItem, chat: ChatClient, model: str) -> ItemRun:
    answer: dict[str, Any] = {"config": cfg.name, "id": item.id, "question": item.question}
    try:
        result = await chat.complete(
            model, closed_book_messages(item), max_tokens=cfg.closed_book.max_tokens, json_mode=False
        )
    except LLMError as exc:
        score = score_closed_book_item(cfg.name, item, None, model=model, error=str(exc))
        return ItemRun(score=score, answer=answer)
    score = score_closed_book_item(
        cfg.name,
        item,
        result.content,
        model=model,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    answer.update(answer=result.content, refused=False, notices=[], citations=[])
    return ItemRun(score=score, answer=answer, responded=True)


def _apply_verdict(score: ItemScore, verdict: Verdict | None, error: str | None) -> None:
    if verdict is None:
        score.judge_rationale = f"judge unavailable: {error}" if error else None
        return
    score.judge_correctness = verdict.correctness
    score.judge_citation_support = verdict.citation_support
    score.judge_refusal_appropriate = verdict.refusal_appropriate
    advice_applies = score.type == "advice_boundary" or "not_advice" in score.expected_notices
    score.judge_advice_ok = verdict.advice_ok if advice_applies else None
    score.judge_rationale = verdict.rationale


async def _judge_item(run: ItemRun, item: GoldenItem, judge: Judge, *, closed_book: bool) -> str | None:
    if not run.responded or "answer" not in run.answer:
        return None
    inp = JudgeInput(
        item=item,
        answer=str(run.answer.get("answer") or ""),
        refused=bool(run.answer.get("refused")),
        refusal_reason=run.answer.get("refusal_reason"),
        notices=run.answer.get("notices") or [],
        citations=run.answer.get("citations") or [],
        closed_book=closed_book,
    )
    verdict, error = await judge.try_judge(inp)
    _apply_verdict(run.score, verdict, error)
    return error


# ---------------------------------------------------------------- one config


async def _run_config(
    cfg: EvalConfig,
    items: Sequence[GoldenItem],
    *,
    service: ServiceClient | None,
    chat_cb: ChatClient | None,
    judge: Judge | None,
    settings: Settings,
    concurrency: int,
) -> tuple[list[ItemRun], dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    closed = cfg.kind == "closed_book"
    model = (cfg.closed_book.model or settings.generation_models[0]) if closed else None
    targets = [i for i in items if i.answerable] if closed else list(items)
    judge_errors: list[str] = []

    async def one(item: GoldenItem) -> ItemRun:
        async with sem:
            if closed:
                assert chat_cb is not None and model is not None
                run = await _closed_book_item(cfg, item, chat_cb, model)
            else:
                assert service is not None
                run = await _service_item(cfg, item, service)
            if judge is not None:
                err = await _judge_item(run, item, judge, closed_book=closed)
                if err:
                    judge_errors.append(err)
            run.score.failure_category = failure_category(run.score, closed_book=closed)
            return run

    runs = list(await asyncio.gather(*(one(i) for i in targets)))
    meta: dict[str, Any] = {
        "kind": cfg.kind,
        "description": cfg.description,
        "baseline": cfg.baseline,
        "source": cfg.source,
        "request": cfg.request.model_dump() if not closed else None,
        "overrides": cfg.overrides,
        "needs_service_support": cfg.needs_service_support,
        "override_honoured": _honoured(cfg, runs),
        "applied_header": sorted({r.applied for r in runs if r.applied}),
        "model": model or _most_common(r.score.model for r in runs),
        "judge_errors": len(judge_errors),
        "judge_error_sample": judge_errors[0] if judge_errors else None,
    }
    return runs, meta


def _honoured(cfg: EvalConfig, runs: Sequence[ItemRun]) -> str:
    if cfg.kind == "closed_book" or not cfg.overrides:
        return "n/a"
    if not any(r.responded for r in runs):
        return "unknown"
    return "yes" if any(r.applied for r in runs) else "no"


def _most_common(values: Any) -> str | None:
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else None


def _failed_items(runs: Sequence[ItemRun]) -> list[dict[str, Any]]:
    return [
        {
            "id": r.score.id,
            "type": r.score.type,
            "language": r.score.language,
            "category": r.score.failure_category,
            "detail": r.score.error or r.score.judge_rationale or "",
        }
        for r in runs
        if r.score.failure_category not in ("ok", "")
    ]


def _golden_summary(path: Path, evaluated: Sequence[GoldenItem]) -> dict[str, Any]:
    everything = read_items(path)

    def count(attr: str, items: Sequence[GoldenItem]) -> dict[str, int]:
        return dict(sorted(Counter(str(getattr(i, attr)) for i in items).items()))

    try:
        shown = path.resolve().relative_to(EVALS.parent).as_posix()
    except ValueError:
        shown = path.as_posix()
    return {
        "path": shown,
        "sha256": golden_sha256(path),
        "total": len(everything),
        "evaluated": len(evaluated),
        "evaluated_ids": [i.id for i in evaluated],
        "by_type": count("type", evaluated),
        "by_language": count("language", evaluated),
        "by_split": count("split", evaluated),
        "by_status": count("review_status", everything),
    }


# ---------------------------------------------------------------- the run


async def run_eval(
    opts: RunOptions,
    *,
    settings: Settings | None = None,
    service_transport: httpx.AsyncBaseTransport | None = None,
    llm_transport: httpx.AsyncBaseTransport | None = None,
    limiter: RateLimiter | None = None,
) -> Path:
    settings = settings or Settings()
    items = load_golden(opts.golden_path, include_drafts=opts.include_drafts, split=opts.split)
    if opts.limit:
        items = items[: opts.limit]
    if not items:
        drafts = "included" if opts.include_drafts else "excluded"
        raise NoItemsError(
            f"no golden items to run (split={opts.split}, drafts {drafts}). "
            "Approve items in evals/golden_set.jsonl, or pass --include-drafts / --split all."
        )
    configs = resolve_configs(opts.configs, opts.config_dir)
    if not configs:
        raise NoItemsError("no configs selected")

    started = datetime.now(UTC)
    name = opts.run_name or started.strftime("%Y%m%d-%H%M%S")
    out_dir = opts.results_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    limiter = limiter or RateLimiter(min(35, settings.llm_max_rpm))

    def chat(sub: str) -> ChatClient:  # judge and closed-book share one key, so one rate limiter
        return ChatClient(
            settings,
            cache_dir=opts.cache_dir / sub,
            use_cache=opts.use_cache,
            offline=opts.offline,
            transport=llm_transport,
            limiter=limiter,
        )

    judge_chat, cb_chat = chat("judge"), chat("closed_book")
    service = ServiceClient(
        opts.base_url,
        cache_dir=opts.cache_dir / "api",
        api_prefix=opts.api_prefix,
        email=opts.email,
        password=opts.password,
        use_cache=opts.use_cache,
        offline=opts.offline,
        transport=service_transport,
    )
    judge = Judge(judge_chat, settings.judge_model) if opts.judge else None

    all_runs: list[ItemRun] = []
    config_metrics: dict[str, Any] = {}
    try:
        for cfg in configs:
            print(f"[eval] {cfg.name}: {len(items)} items ({cfg.kind})", flush=True)
            t0 = time.perf_counter()
            runs, meta = await _run_config(
                cfg,
                items,
                service=service,
                chat_cb=cb_chat,
                judge=judge,
                settings=settings,
                concurrency=opts.concurrency,
            )
            meta["seconds"] = round(time.perf_counter() - t0, 1)
            scores = [r.score for r in runs]
            config_metrics[cfg.name] = {
                "meta": meta,
                **breakdown(scores),
                "failed_items": _failed_items(runs),
            }
            if meta["override_honoured"] == "no":
                print(
                    f"[eval] WARNING {cfg.name}: the service ignored X-Eval-Config; "
                    "this row ran on the service defaults",
                    flush=True,
                )
            all_runs.extend(runs)
        if opts.ragas:
            await _maybe_ragas(config_metrics, configs, all_runs, items, settings)
    finally:
        await service.aclose()
        await judge_chat.aclose()
        await cb_chat.aclose()

    metrics: dict[str, Any] = {
        "run": {
            "name": name,
            "started_at": started.isoformat(timespec="seconds"),
            "base_url": opts.base_url,
            "split": opts.split,
            "include_drafts": opts.include_drafts,
            "limit": opts.limit,
            "offline": opts.offline,
            "use_cache": opts.use_cache,
            "judge": opts.judge,
            "judge_model": settings.judge_model,
            "judge_prompt_version": PROMPT_VERSION,
            "generation_models": settings.generation_models,
            "configs": [c.name for c in configs],
            "command": opts.command or "python -m evals.run",
            "network_calls": {
                "service": service.network_calls,
                "judge": judge_chat.network_calls,
                "closed_book": cb_chat.network_calls,
            },
            "golden": _golden_summary(opts.golden_path, items),
        },
        "configs": config_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report.write_per_question_csv(out_dir / "per_question.csv", [r.score for r in all_runs])
    with (out_dir / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_runs:
            fh.write(json.dumps(r.answer, ensure_ascii=False) + "\n")
    (out_dir / "report.md").write_text(report.render_report(metrics), encoding="utf-8")
    if opts.eval_md is not None:
        blocks = report.update_eval_md(opts.eval_md, metrics)
        print(
            f"[eval] EVAL.md blocks refreshed: {', '.join(blocks) or 'none (no markers found)'}", flush=True
        )
    print(f"[eval] wrote {out_dir}", flush=True)
    return out_dir


async def _maybe_ragas(
    config_metrics: dict[str, Any],
    configs: Sequence[EvalConfig],
    runs: Sequence[ItemRun],
    items: Sequence[GoldenItem],
    settings: Settings,
) -> None:
    from evals import ragas_eval

    if not ragas_eval.is_available():
        print("[eval] RAGAS not installed (uv pip install -r requirements-eval.in); skipped", flush=True)
        return
    by_id = {i.id: i for i in items}
    for cfg in configs:
        if cfg.kind != "service":
            continue
        records = [r.answer for r in runs if r.score.config == cfg.name and r.responded]
        config_metrics[cfg.name]["ragas"] = await ragas_eval.score(records, by_id, settings)


def main(argv: Sequence[str] | None = None) -> int:
    opts = parse_args(argv)
    try:
        asyncio.run(run_eval(opts))
    except (NoItemsError, FileNotFoundError) as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
