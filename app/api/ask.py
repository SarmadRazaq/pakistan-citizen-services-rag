from __future__ import annotations

from fastapi import APIRouter, status

from app.api._stub import not_implemented
from app.api.deps import TenantDep
from app.api.responses import problems
from app.schemas import AskRequest, AskResponse, FeedbackOut, FeedbackRequest, SearchRequest, SearchResponse

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Full RAG answer with citations and every retrieved chunk (refusals are 200)",
    responses=problems(400, 401, 404, 422, 429, 503),
)
async def ask(tenant: TenantDep, body: AskRequest) -> AskResponse:
    not_implemented("A6")


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieval only: ranked chunks with every score",
    responses=problems(401, 404, 422, 503),
)
async def search(tenant: TenantDep, body: SearchRequest) -> SearchResponse:
    not_implemented("A6")


@router.post(
    "/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
    summary="Thumbs up/down on one of my own answers",
    responses=problems(401, 404, 422),
)
async def feedback(tenant: TenantDep, body: FeedbackRequest) -> FeedbackOut:
    not_implemented("A6")
