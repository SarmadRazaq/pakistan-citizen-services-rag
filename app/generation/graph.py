"""The ask pipeline as a LangGraph `StateGraph` (blueprint §6).

    guard_input → corpus_check → understand → retrieve → relevance_gate → generate ⇄ guard_output → finalize
                       │              │                        │                         │
                       └──────────────┴────────── refuse ──────┴─────────────────────────┘

Conditional edges are the refusal paths (no_documents, off_topic, insufficient_context,
guardrail_output). `guard_output` loops back to `generate` for the JSON repair retry and the single
regeneration. An `AppError` raised inside a node (400 input guard, 503 LLM unavailable) is stored
in the state, the graph ends, and the service re-raises it. Every node records its latency.
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Annotated, Any, Literal, Protocol, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph
from sqlalchemy import func, select

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.eval_overrides import EvalOverrides
from app.core.tenancy import TenantContext
from app.db.models import Document
from app.db.session import Database
from app.generation.answer import AnswerCheck, CheckKind, VerifiedCitation, evaluate_answer
from app.generation.prompts import (
    CANARY_TOKEN,
    ContextSource,
    advice_retry_instruction,
    build_answer_messages,
    build_repair_messages,
    build_system_prompt,
    citation_retry_instruction,
    language_retry_instruction,
    message,
)
from app.generation.query_understanding import QueryUnderstanding, detect_language, understand
from app.guardrails.advice import Referral, match_emergency_referrals, match_referrals
from app.guardrails.emergency import is_emergency
from app.guardrails.input_guards import check_question
from app.guardrails.output_guards import DatedSource, advice_notices, dated_notices, emergency_notices
from app.protocols import (
    Candidate,
    InjectionClassifier,
    LLMClient,
    LLMMessage,
    LLMResult,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    SearchFilters,
)
from app.schemas import AnswerLanguage, AskRequest, DocumentStatus, LanguageCode, Notice, RefusalReason

log = structlog.get_logger(__name__)

GENERATE_MAX_TOKENS = 1200
MAX_GENERATE_CALLS = 3  # first answer + one JSON repair + one regeneration
SERVABLE_STATUSES = (
    DocumentStatus.READY.value,
    DocumentStatus.READY_STALE.value,
    DocumentStatus.UPDATING.value,
)

RetryMode = Literal["repair", "citations", "language", "advice"]


def _sum_timings(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    out = dict(left)
    for key, value in right.items():
        out[key] = out.get(key, 0) + value
    return out


class AskState(TypedDict, total=False):
    # inputs
    ctx: TenantContext
    request: AskRequest
    overrides: EvalOverrides | None  # eval-only (X-Eval-Config); None in normal serving
    # accumulated
    flags: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, int], _sum_timings]
    llm_calls: Annotated[list[LLMResult], operator.add]
    # per node
    error: AppError | None
    question: str
    guard_input: str
    emergency: bool
    understanding: QueryUnderstanding
    answer_language: LanguageCode
    filters: SearchFilters
    retrieval: RetrievalResult
    llm_used: Literal["hosted", "local"]
    private_sent_hosted: bool
    base_messages: list[LLMMessage]
    raw: str
    generate_calls: int
    repair_used: bool
    regen_used: bool
    retry_mode: RetryMode | None
    check: AnswerCheck
    refuse: bool
    refusal_reason: RefusalReason | None
    guard_output: Literal["passed", "repaired", "refused"]
    answer: str
    answerable: bool
    citations: tuple[VerifiedCitation, ...]
    unanswered_parts: list[str]
    notices: list[Notice]


@dataclass(frozen=True, slots=True)
class PipelineDeps:
    settings: Settings
    db: Database
    retriever: Retriever
    llm_hosted: LLMClient
    llm_local: LLMClient | None
    referrals: tuple[Referral, ...] = ()
    classifier: InjectionClassifier | None = None
    clock: Callable[[], date] = date.today


class Node(Protocol):
    def __call__(self, state: AskState) -> Awaitable[dict[str, Any]]: ...


def _timed(name: str, fn: Node) -> Node:
    async def run(state: AskState) -> dict[str, Any]:
        started = perf_counter()
        try:
            update = await fn(state)
        except AppError as exc:
            update = {"error": exc}
        update["timings"] = {name: int((perf_counter() - started) * 1000)}
        return update

    return run


def answer_language_for(request: AskRequest, question_language: LanguageCode) -> LanguageCode:
    if request.answer_language is AnswerLanguage.AUTO:
        return question_language
    return LanguageCode(request.answer_language.value)


def context_sources(ctx: TenantContext, context: tuple[Candidate, ...]) -> list[ContextSource]:
    """S{i} ↔ context[i-1], by construction."""
    sources = []
    for i, cand in enumerate(context, start=1):
        p = cand.point.payload
        sources.append(
            ContextSource(
                marker=f"S{i}",
                text=p.text,
                scope="public" if cand.point.owner_id == ctx.public_owner_id else "private",
                agency=p.agency,
                jurisdiction=p.jurisdiction,
                date=p.effective_date.isoformat() if p.effective_date else None,
                file=p.file_name,
                page=p.page_start,
            )
        )
    return sources


def has_private(ctx: TenantContext, context: tuple[Candidate, ...]) -> bool:
    return any(c.point.owner_id != ctx.public_owner_id for c in context)


def build_graph(deps: PipelineDeps) -> Any:
    settings = deps.settings

    # ------------------------------------------------------------ nodes

    async def guard_input(state: AskState) -> dict[str, Any]:
        result = await check_question(
            state["request"].question, max_chars=settings.max_question_chars, classifier=deps.classifier
        )
        return {
            "question": result.text,
            "guard_input": result.status,
            "flags": list(result.flags),
            "emergency": is_emergency(result.text),
        }

    def emergency_for(state: AskState, language: LanguageCode) -> list[Notice]:
        """GO-11: the emergency Notices (generic first) when the guard is on and the flag is set."""
        if not (settings.emergency_guard and state.get("emergency")):
            return []
        q = state.get("understanding")
        places = {j.value for j in q.jurisdictions} if q else set()
        return emergency_notices(language, match_emergency_referrals(deps.referrals, jurisdictions=places))

    async def corpus_check(state: AskState) -> dict[str, Any]:
        ctx, request = state["ctx"], state["request"]
        stmt = select(func.count(Document.id)).where(Document.status.in_(SERVABLE_STATUSES))
        if request.doc_ids:
            stmt = stmt.where(Document.id.in_(request.doc_ids))
        async with deps.db.session(ctx) as s:
            count = int((await s.scalar(stmt)) or 0)
        if count == 0:
            return {"refuse": True, "refusal_reason": RefusalReason.NO_DOCUMENTS}
        return {}

    async def understand_node(state: AskState) -> dict[str, Any]:
        q = await understand(state["question"], deps.llm_hosted)
        update: dict[str, Any] = {
            "understanding": q,
            "answer_language": answer_language_for(state["request"], q.language),
            "flags": list(q.flags),
            "emergency": state.get("emergency", False) or q.emergency,
        }
        if q.off_topic:
            update |= {"refuse": True, "refusal_reason": RefusalReason.OFF_TOPIC}
        return update

    async def retrieve(state: AskState) -> dict[str, Any]:
        ctx, request, q = state["ctx"], state["request"], state["understanding"]
        req_agencies = tuple(request.agencies) if request.agencies else None
        req_places = tuple(j.value for j in request.jurisdictions) if request.jurisdictions else None
        detected_agencies = None if req_agencies else (q.agencies or None)
        detected_places = None
        if req_places is None and q.jurisdictions:
            detected_places = tuple(dict.fromkeys([*(j.value for j in q.jurisdictions), "federal"]))
        doc_ids = tuple(request.doc_ids) if request.doc_ids else None
        context_k = request.top_k if "top_k" in request.model_fields_set else settings.context_k

        async def run(agencies: tuple[str, ...] | None, places: tuple[str, ...] | None) -> RetrievalResult:
            query = RetrievalQuery(
                variants=q.variants or (state["question"],),
                filters=SearchFilters(doc_ids=doc_ids, agencies=agencies, jurisdictions=places),
                candidates_k=settings.candidates_k,
                context_k=context_k,
                overrides=state.get("overrides"),
            )
            return await deps.retriever.retrieve(ctx, query)

        agencies, places = req_agencies or detected_agencies, req_places or detected_places
        result = await run(agencies, places)
        flags: list[str] = []
        weak = not result.context or (
            result.max_rerank is not None and result.max_rerank < settings.tau_refuse
        )
        if weak and detected_agencies:
            # A detected (not requested) agency filter can hide untagged documents: retry without it.
            relaxed = await run(req_agencies, places)
            if (relaxed.max_rerank or 0.0) > (result.max_rerank or 0.0) or (
                relaxed.context and not result.context
            ):
                result, agencies = relaxed, req_agencies
                flags.append("agency_filter_relaxed")
        if not result.candidates and detected_places:
            # Nothing at all in the detected province: fall back to every province (answered per province).
            relaxed = await run(agencies, req_places)
            if relaxed.candidates:
                result, places = relaxed, req_places
                flags.append("jurisdiction_filter_relaxed")
        return {
            "retrieval": result,
            "filters": SearchFilters(doc_ids=doc_ids, agencies=agencies, jurisdictions=places),
            "flags": flags,
        }

    async def relevance_gate(state: AskState) -> dict[str, Any]:
        result = state["retrieval"]
        below = result.max_rerank is not None and result.max_rerank < settings.tau_refuse
        if not result.context or below:
            return {"refuse": True, "refusal_reason": RefusalReason.INSUFFICIENT_CONTEXT}
        return {}

    async def generate(state: AskState) -> dict[str, Any]:
        ctx, result = state["ctx"], state["retrieval"]
        language = state["answer_language"]
        sources = context_sources(ctx, result.context)
        update: dict[str, Any] = {}
        if "llm_used" not in state:
            private = has_private(ctx, result.context)
            if private and settings.private_context_policy == "local_only":
                if deps.llm_local is None:
                    raise AppError(
                        ErrorCode.LLM_UNAVAILABLE, "The local model for private documents is not configured."
                    )
                update |= {
                    "llm_used": "local",
                    "private_sent_hosted": False,
                    "flags": ["private_context_local"],
                }
            else:
                update |= {"llm_used": "hosted", "private_sent_hosted": private}
        llm_used = update.get("llm_used") or state["llm_used"]
        llm = deps.llm_local if llm_used == "local" and deps.llm_local is not None else deps.llm_hosted

        mode = state.get("retry_mode")
        base = state.get("base_messages") or build_answer_messages(state["question"], sources, language)
        if mode == "repair":
            messages = build_repair_messages(base, state.get("raw", ""))
        elif mode == "citations":
            messages = build_answer_messages(
                state["question"], sources, language, extra_instruction=citation_retry_instruction()
            )
        elif mode == "language":
            messages = build_answer_messages(
                state["question"], sources, language, extra_instruction=language_retry_instruction(language)
            )
        elif mode == "advice":
            messages = build_answer_messages(
                state["question"], sources, language, extra_instruction=advice_retry_instruction()
            )
        else:
            messages = base
        overrides = state.get("overrides")
        if overrides and overrides.generation_model:  # eval-only: restrict the chain to that model
            reply = await llm.complete(
                messages,
                purpose="generate",
                max_tokens=GENERATE_MAX_TOKENS,
                model=overrides.generation_model,
            )
        else:
            reply = await llm.complete(messages, purpose="generate", max_tokens=GENERATE_MAX_TOKENS)
        return update | {
            "base_messages": base,
            "raw": reply.text,
            "llm_calls": [reply],
            "generate_calls": state.get("generate_calls", 0) + 1,
            "retry_mode": None,
        }

    async def guard_output(state: AskState) -> dict[str, Any]:
        ctx, result = state["ctx"], state["retrieval"]
        language = state["answer_language"]
        sources = {s.marker: s.text for s in context_sources(ctx, result.context)}
        check = evaluate_answer(
            state["raw"],
            sources,
            system_prompt=build_system_prompt(language),
            canary=CANARY_TOKEN,
            answer_language=language,
            min_score=settings.quote_match_min,
            advice_request=state["understanding"].advice_request,
        )
        update: dict[str, Any] = {"check": check, "flags": list(check.flags)}
        calls_left = state.get("generate_calls", 0) < MAX_GENERATE_CALLS
        regen_left = calls_left and not state.get("regen_used", False)
        refuse = {"refuse": True, "refusal_reason": RefusalReason.GUARDRAIL_OUTPUT, "guard_output": "refused"}

        if check.kind is CheckKind.LEAK:
            log.warning("security.output_leak_blocked", kind=check.leak, request_id=ctx.request_id)
            return update | refuse
        if check.kind is CheckKind.MALFORMED:
            if calls_left and not state.get("repair_used", False):
                return update | {"retry_mode": "repair", "repair_used": True}
            return update | refuse
        if check.kind is CheckKind.UNANSWERABLE:
            return update | {"refuse": True, "refusal_reason": RefusalReason.INSUFFICIENT_CONTEXT}
        if check.kind is CheckKind.CITATIONS:
            if regen_left:
                return update | {"retry_mode": "citations", "regen_used": True}
            if check.verification is None or not check.verification.citations:
                return update | refuse  # GO-2: still no valid citation
            # Some claims kept valid citations: serve them, with the unverifiable markers removed.
            return update | {"guard_output": "repaired"}
        if check.kind is CheckKind.LANGUAGE and regen_left:
            return update | {"retry_mode": "language", "regen_used": True}
        if check.kind is CheckKind.DIRECTIVE_ADVICE and regen_left:
            return update | {"retry_mode": "advice", "regen_used": True}
        retried = state.get("generate_calls", 0) > 1
        changed = "secret_redacted" in check.flags or "uncited_marker" in check.flags
        return update | {"guard_output": "repaired" if retried or changed else "passed"}

    async def finalize(state: AskState) -> dict[str, Any]:
        check, q = state["check"], state["understanding"]
        language = state["answer_language"]
        citations = check.verification.citations if check.verification else ()
        context = state["retrieval"].context
        cited = [context[int(c.marker[1:]) - 1] for c in citations]
        notices = emergency_for(state, language)
        if q.advice_request and settings.advice_guard:
            topics = {c.point.payload.topic or "" for c in cited}
            places = {j.value for j in q.jurisdictions} | {c.point.payload.jurisdiction or "" for c in cited}
            notices += advice_notices(
                language, match_referrals(deps.referrals, topics=topics, jurisdictions=places)
            )
        notices += dated_notices(
            (
                DatedSource(
                    doc_id=c.point.payload.doc_id,
                    doc_type=c.point.payload.doc_type,
                    effective_date=c.point.payload.effective_date,
                    file_name=c.point.payload.file_name,
                    source_url=c.point.payload.source_url,
                )
                for c in cited
            ),
            today=deps.clock(),
            stale_after_months=settings.stale_after_months,
            language=language,
        )
        return {
            "answer": check.answer,
            "answerable": True,
            "citations": citations,
            "unanswered_parts": list(check.unanswered_parts),
            "refusal_reason": None,
            "notices": notices,
            "guard_output": state.get("guard_output", "passed"),
        }

    async def refuse(state: AskState) -> dict[str, Any]:
        reason = state.get("refusal_reason") or RefusalReason.INSUFFICIENT_CONTEXT
        language = state.get("answer_language") or answer_language_for(
            state["request"], detect_language(state.get("question") or state["request"].question)
        )
        check = state.get("check")
        return {
            "answer": message(reason.value, language),
            "answerable": False,
            "citations": (),
            "unanswered_parts": list(check.unanswered_parts) if check else [],
            "answer_language": language,
            "notices": emergency_for(state, language),
        }

    # ------------------------------------------------------------ edges

    def after(next_node: str) -> Callable[[AskState], str]:
        def route(state: AskState) -> str:
            if state.get("error") is not None:
                return END
            if state.get("refuse"):
                return "refuse"
            return next_node

        return route

    def after_guard_output(state: AskState) -> str:
        if state.get("refuse"):
            return "refuse"
        if state.get("retry_mode"):
            return "generate"
        return "finalize"

    graph = StateGraph(AskState)
    nodes: Mapping[str, Node] = {
        "guard_input": guard_input,
        "corpus_check": corpus_check,
        "understand": understand_node,
        "retrieve": retrieve,
        "relevance_gate": relevance_gate,
        "generate": generate,
        "guard_output": guard_output,
        "finalize": finalize,
        "refuse": refuse,
    }
    for name, fn in nodes.items():
        graph.add_node(name, _timed(name, fn))

    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", after("corpus_check"), ["corpus_check", "refuse", END])
    graph.add_conditional_edges("corpus_check", after("understand"), ["understand", "refuse", END])
    graph.add_conditional_edges("understand", after("retrieve"), ["retrieve", "refuse", END])
    graph.add_conditional_edges("retrieve", after("relevance_gate"), ["relevance_gate", "refuse", END])
    graph.add_conditional_edges("relevance_gate", after("generate"), ["generate", "refuse", END])
    graph.add_conditional_edges("generate", after("guard_output"), ["guard_output", "refuse", END])
    graph.add_conditional_edges("guard_output", after_guard_output, ["generate", "finalize", "refuse"])
    graph.add_edge("finalize", END)
    graph.add_edge("refuse", END)
    return graph.compile()
