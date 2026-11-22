"""Thin handlers for /ask, /search and /feedback. All logic lives in the AskService (app/generation).

The service is looked up inside the handler (not as a `Depends`) so that body validation (422)
still runs before the "service not wired" 501, exactly as the contract tests expect: FastAPI
resolves dependencies before it reports body errors.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from app.api.deps import TenantDep, get_ask_service
from app.api.responses import problems
from app.schemas import AskRequest, AskResponse, FeedbackOut, FeedbackRequest, SearchRequest, SearchResponse

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Full RAG answer with citations and every retrieved chunk (refusals are 200)",
    responses=problems(400, 401, 404, 422, 429, 503),
)
async def ask(request: Request, tenant: TenantDep, body: AskRequest) -> AskResponse:
    return await get_ask_service(request).ask(tenant, body)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieval only: ranked chunks with every score",
    responses=problems(401, 404, 422, 503),
)
async def search(request: Request, tenant: TenantDep, body: SearchRequest) -> SearchResponse:
    return await get_ask_service(request).search(tenant, body)


@router.post(
    "/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
    summary="Thumbs up/down on one of my own answers",
    responses=problems(401, 404, 422),
)
async def feedback(request: Request, tenant: TenantDep, body: FeedbackRequest) -> FeedbackOut:
    return await get_ask_service(request).feedback(tenant, body)
