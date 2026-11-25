"""`RagAskService`: the `AskService` behind /ask, /search and /feedback (blueprint §6, §8.4).

- `ask` narrows the tenant to the request scope, checks `doc_ids` (GI-9), runs the LangGraph
  pipeline, builds the full `AskResponse` (every fetched candidate, not only the context) and
  writes one `RequestLog` row without the question text.
- `search` is retrieval only.
- `feedback` is accepted only on the caller's own `request_id` (another user's → 404, E38).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import date
from time import perf_counter
from typing import Any

import structlog
from sqlalchemy import select

from app.config import Settings
from app.core.context import get_request_id
from app.core.errors import AppError, ErrorCode
from app.core.eval_overrides import EvalOverrides
from app.core.ids import new_feedback_id
from app.core.tenancy import TenantContext, Visibility
from app.db.models import Document, Feedback, RequestLog, utcnow
from app.db.session import Database
from app.generation.answer import VerifiedCitation
from app.generation.graph import AskState, PipelineDeps, build_graph
from app.generation.prompts import PROMPT_VERSION
from app.generation.query_understanding import QueryUnderstanding, detect_language
from app.guardrails.advice import Referral, load_referrals
from app.guardrails.input_guards import sanitize
from app.guardrails.output_guards import third_party_notice
from app.observability.cost import ModelUsage, request_cost
from app.protocols import (
    Candidate,
    InjectionClassifier,
    LLMClient,
    LLMResult,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    SearchFilters,
)
from app.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    FeedbackOut,
    FeedbackRequest,
    GuardrailInfo,
    Jurisdiction,
    LanguageCode,
    LanguageInfo,
    Outcome,
    ProviderInfo,
    RetrievalInfo,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    Usage,
)

log = structlog.get_logger(__name__)


def _enum_or_none[E: (Jurisdiction, LanguageCode)](enum: type[E], value: str | None) -> E | None:
    if not value:
        return None
    try:
        return enum(value)
    except ValueError:
        return None


def outcome_for(*, answerable: bool, degraded: bool, unanswered_parts: list[str]) -> Outcome:
    """GLOSSARY.md "Outcomes": a refusal wins, then degraded, then partial."""
    if not answerable:
        return Outcome.REFUSAL
    if degraded:
        return Outcome.DEGRADED_ANSWER
    if unanswered_parts:
        return Outcome.PARTIAL_ANSWER
    return Outcome.ANSWER


def _scope(ctx: TenantContext, owner_id: str) -> Visibility:
    return Visibility.PUBLIC if owner_id == ctx.public_owner_id else Visibility.PRIVATE


def retrieved_chunks(
    ctx: TenantContext, result: RetrievalResult | None, cited_ids: set[str] | None = None
) -> list[RetrievedChunk]:
    """Every fetched candidate (not only the context), with all scores and used/cited flags."""
    if result is None:
        return []
    cited_ids = cited_ids or set()
    context_ids = {c.point.point_id for c in result.context}
    chunks = []
    for cand in result.candidates:
        p, point = cand.point.payload, cand.point
        chunks.append(
            RetrievedChunk(
                rank=cand.rank,
                chunk_id=point.point_id,
                document_id=p.doc_id,
                scope=_scope(ctx, point.owner_id),
                file_name=p.file_name,
                agency=p.agency,
                jurisdiction=_enum_or_none(Jurisdiction, p.jurisdiction),
                effective_date=p.effective_date,
                page=p.page_start,
                page_end=p.page_end,
                section_path=list(p.section_path),
                chunk_index=p.chunk_index,
                language=_enum_or_none(LanguageCode, p.language),
                is_table=p.is_table,
                is_ocr=p.is_ocr,
                text=p.text,
                similarity=point.similarity,
                sparse_score=point.sparse_score,
                rrf_score=point.rrf_score,
                rerank_score=cand.rerank_score,
                used_in_context=cand.used_in_context or point.point_id in context_ids,
                cited=point.point_id in cited_ids,
                injection_flag=p.injection_flag,
            )
        )
    return chunks


def build_citation(ctx: TenantContext, marker: str, cand: Candidate, quote: str) -> Citation:
    p = cand.point.payload
    return Citation(
        marker=marker,
        chunk_id=cand.point.point_id,
        document_id=p.doc_id,
        scope=_scope(ctx, cand.point.owner_id),
        file_name=p.file_name,
        agency=p.agency,
        jurisdiction=_enum_or_none(Jurisdiction, p.jurisdiction),
        effective_date=p.effective_date,
        page=p.page_start,
        section=" › ".join(p.section_path) or None,
        source_url=p.source_url,
        quote=quote,
    )


class RagAskService:
    """Implements `app.protocols.AskService`."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        retriever: Retriever,
        llm_hosted: LLMClient,
        llm_local: LLMClient | None = None,
        *,
        classifier: InjectionClassifier | None = None,
        referrals: Sequence[Referral] | None = None,
        clock: Callable[[], date] = date.today,
    ) -> None:
        self.settings = settings
        self.db = db
        self.retriever = retriever
        if referrals is None:
            referrals = load_referrals(settings.referrals_file.resolve())
        self.deps = PipelineDeps(
            settings=settings,
            db=db,
            retriever=retriever,
            llm_hosted=llm_hosted,
            llm_local=llm_local,
            referrals=tuple(referrals),
            classifier=classifier,
            clock=clock,
        )
        self._graph = build_graph(self.deps)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _request_id(ctx: TenantContext) -> str:
        return ctx.request_id or get_request_id() or uuid.uuid4().hex

    async def _check_doc_ids(self, ctx: TenantContext, doc_ids: Sequence[str] | None) -> None:
        """GI-9: every requested doc id must be readable in this scope; otherwise 404 (never 403)."""
        if not doc_ids:
            return
        wanted = set(doc_ids)
        async with self.db.session(ctx) as s:
            found = set((await s.scalars(select(Document.id).where(Document.id.in_(wanted)))).all())
        if wanted - found:
            raise AppError(ErrorCode.NOT_FOUND, "Document not found.")

    async def _log_request(
        self,
        ctx: TenantContext,
        request_id: str,
        *,
        route: str,
        status: int,
        started: float,
        error_code: str | None = None,
        llm: LLMResult | None = None,
        degraded: bool = False,
        answerable: bool | None = None,
        tokens: tuple[int, int] = (0, 0),
        calls: Sequence[ModelUsage] = (),
    ) -> None:
        cost_usd, list_price_usd = request_cost(self.settings, calls)
        row = RequestLog(
            request_id=request_id,
            user_id=ctx.user_id,
            route=route,
            status=status,
            latency_ms=int((perf_counter() - started) * 1000),
            error_code=error_code,
            llm_endpoint=llm.endpoint if llm else None,
            llm_model=llm.model if llm else None,
            degraded=degraded,
            answerable=answerable,
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            cost_usd=cost_usd,
            list_price_usd=list_price_usd,
        )
        try:
            async with self.db.session(ctx) as s:
                s.add(row)
        except Exception as exc:  # the log must never turn an answer into an error
            log.warning("request_log.write_failed", error=type(exc).__name__, route=route)

    def _model_usage(self, state: AskState) -> list[ModelUsage]:
        """Every model call of the request (helper + each generation attempt) for cost accounting."""
        q: QueryUnderstanding | None = state.get("understanding")
        calls = [
            ModelUsage(r.model, r.endpoint, r.input_tokens, r.output_tokens)
            for r in state.get("llm_calls", [])
        ]
        if q and any(q.usage):
            calls.insert(0, ModelUsage(self.settings.helper_model, "nvidia", q.usage[0], q.usage[1]))
        return calls

    # ------------------------------------------------------------ ask

    async def ask(
        self, ctx: TenantContext, request: AskRequest, *, overrides: EvalOverrides | None = None
    ) -> AskResponse:
        started = perf_counter()
        request_id = self._request_id(ctx)
        ctx = ctx.with_scope(request.scope)
        if self.settings.log_question_text:
            log.debug("ask.question", question=request.question)
        try:
            await self._check_doc_ids(ctx, request.doc_ids)
            initial: AskState = {
                "ctx": ctx,
                "request": request,
                "overrides": overrides,
                "flags": [],
                "timings": {},
                "llm_calls": [],
            }
            state: AskState = await self._graph.ainvoke(initial)
            error = state.get("error")
            if error is not None:
                raise error
        except AppError as exc:
            await self._log_request(
                ctx,
                request_id,
                route="ask",
                status=exc.code.status,
                started=started,
                error_code=exc.code.slug,
            )
            raise
        response = self._build_response(ctx, request, request_id, state, started)
        await self._log_request(
            ctx,
            request_id,
            route="ask",
            status=200,
            started=started,
            llm=state["llm_calls"][-1] if state.get("llm_calls") else None,
            degraded=response.degraded,
            answerable=response.answerable,
            tokens=(response.usage.input_tokens, response.usage.output_tokens),
            calls=self._model_usage(state),
        )
        log.info(
            "ask.done",
            prompt_version=PROMPT_VERSION,
            answerable=response.answerable,
            refusal_reason=response.refusal_reason,
            guard_output=response.guardrails.output,
            flags=response.guardrails.flags,
            latency_ms=response.latency_ms.get("total"),
            steps_ms=response.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response

    def _build_response(
        self, ctx: TenantContext, request: AskRequest, request_id: str, state: AskState, started: float
    ) -> AskResponse:
        q: QueryUnderstanding | None = state.get("understanding")
        result = state.get("retrieval")
        context = result.context if result else ()
        verified: tuple[VerifiedCitation, ...] = state.get("citations", ())
        citations = [build_citation(ctx, c.marker, context[int(c.marker[1:]) - 1], c.quote) for c in verified]
        cited_ids = {c.chunk_id for c in citations}

        question_language = q.language if q else detect_language(state.get("question") or request.question)
        answer_language = state.get("answer_language") or question_language

        generate_calls = list(state.get("llm_calls", []))
        helper_tokens = q.usage if q else (0, 0)
        input_tokens = helper_tokens[0] + sum(r.input_tokens for r in generate_calls)
        output_tokens = helper_tokens[1] + sum(r.output_tokens for r in generate_calls)
        last = generate_calls[-1] if generate_calls else None
        provider = (
            ProviderInfo(
                endpoint=last.endpoint, model=last.model, seller=last.seller, attempts=len(generate_calls)
            )
            if last
            else ProviderInfo(endpoint="none")
        )
        degraded = state.get("llm_used") == "local" or any(r.degraded for r in generate_calls)

        notices = list(state.get("notices", []))
        if state.get("private_sent_hosted"):
            notices.append(third_party_notice(answer_language))

        filters: SearchFilters | None = state.get("filters")
        flags = list(dict.fromkeys(state.get("flags", [])))
        timings = dict(state.get("timings", {}))
        if result is not None:
            for key, value in result.latency_ms.items():
                timings[f"retrieve.{key}"] = value
        timings["total"] = int((perf_counter() - started) * 1000)
        retrieval = RetrievalInfo(
            query_variants=list(q.variants) if q else [state.get("question") or request.question],
            sub_queries=list(q.sub_queries) if q else [],
            agency_filter=list(filters.agencies or ()) if filters else [],
            jurisdiction_filter=[
                j for j in (_enum_or_none(Jurisdiction, v) for v in (filters.jurisdictions or ())) if j
            ]
            if filters
            else [],
            advice_request=bool(q and q.advice_request),
            candidates_fetched=len(result.candidates) if result else 0,
            chunks_used=len(context),
            threshold=self.settings.tau_refuse,
            chunks=retrieved_chunks(ctx, result, cited_ids),
        )
        answerable = bool(state.get("answerable", False))
        unanswered_parts = state.get("unanswered_parts", [])
        return AskResponse(
            request_id=request_id,
            answer=state.get("answer", ""),
            answerable=answerable,
            outcome=outcome_for(answerable=answerable, degraded=degraded, unanswered_parts=unanswered_parts),
            refusal_reason=state.get("refusal_reason") if not answerable else None,
            unanswered_parts=unanswered_parts,
            degraded=degraded,
            notices=notices,
            language=LanguageInfo(question=question_language, answer=answer_language),
            citations=citations,
            retrieval=retrieval,
            guardrails=GuardrailInfo(
                input="sanitized" if state.get("guard_input") == "sanitized" else "passed",
                output=state.get("guard_output", "passed"),
                flags=flags,
            ),
            provider=provider,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            latency_ms=timings,
        )

    # ------------------------------------------------------------ search

    async def search(
        self, ctx: TenantContext, request: SearchRequest, *, overrides: EvalOverrides | None = None
    ) -> SearchResponse:
        started = perf_counter()
        request_id = self._request_id(ctx)
        ctx = ctx.with_scope(request.scope)
        try:
            query, _ = sanitize(request.query.strip())
            query = query.strip()
            if not query:
                raise AppError(ErrorCode.QUESTION_EMPTY, "The query is empty.")
            if len(query) > self.settings.max_question_chars:
                raise AppError(ErrorCode.QUESTION_TOO_LONG, "The query is too long.")
            await self._check_doc_ids(ctx, request.doc_ids)
            filters = SearchFilters(
                doc_ids=tuple(request.doc_ids) if request.doc_ids else None,
                agencies=tuple(request.agencies) if request.agencies else None,
                jurisdictions=tuple(j.value for j in request.jurisdictions)
                if request.jurisdictions
                else None,
            )
            t0 = perf_counter()
            result = await self.retriever.retrieve(
                ctx,
                RetrievalQuery(
                    variants=(query,),
                    filters=filters,
                    candidates_k=request.top_k,
                    context_k=min(self.settings.context_k, request.top_k),
                    overrides=overrides,
                ),
            )
            retrieve_ms = int((perf_counter() - t0) * 1000)
        except AppError as exc:
            await self._log_request(
                ctx,
                request_id,
                route="search",
                status=exc.code.status,
                started=started,
                error_code=exc.code.slug,
            )
            raise
        latency: dict[str, Any] = {f"retrieve.{k}": v for k, v in result.latency_ms.items()}
        latency |= {"retrieve": retrieve_ms, "total": int((perf_counter() - started) * 1000)}
        response = SearchResponse(
            request_id=request_id,
            retrieval=RetrievalInfo(
                query_variants=[query],
                agency_filter=list(filters.agencies or ()),
                jurisdiction_filter=list(request.jurisdictions or []),
                candidates_fetched=len(result.candidates),
                chunks_used=len(result.context),
                threshold=self.settings.tau_refuse,
                chunks=retrieved_chunks(ctx, result),
            ),
            latency_ms=latency,
        )
        await self._log_request(ctx, request_id, route="search", status=200, started=started)
        return response

    # ------------------------------------------------------------ feedback

    async def feedback(self, ctx: TenantContext, request: FeedbackRequest) -> FeedbackOut:
        async with self.db.session(ctx) as s:
            owned = (
                await s.scalars(
                    select(RequestLog.id).where(
                        RequestLog.request_id == request.request_id, RequestLog.user_id == ctx.user_id
                    )
                )
            ).first()
            if owned is None:
                raise AppError(ErrorCode.NOT_FOUND, "Request not found.")
            row = Feedback(
                id=new_feedback_id(),
                user_id=ctx.user_id,
                request_id=request.request_id,
                rating=request.rating,
                comment=request.comment,
                created_at=utcnow(),
            )
            s.add(row)
        return FeedbackOut(
            id=row.id, request_id=row.request_id, rating=request.rating, created_at=row.created_at
        )


def build_ask_service(
    settings: Settings,
    db: Database,
    retriever: Retriever,
    llm_hosted: LLMClient,
    llm_local: LLMClient | None = None,
    *,
    classifier: InjectionClassifier | None = None,
    referrals: Sequence[Referral] | None = None,
) -> RagAskService:
    """Factory for app/main.py: `app.state.ask = build_ask_service(settings, db, retriever, llm, local)`."""
    return RagAskService(
        settings, db, retriever, llm_hosted, llm_local, classifier=classifier, referrals=referrals
    )
