from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import TenantDep
from app.api.responses import problems
from app.core.auth_service import AuthService
from app.core.errors import AppError, ErrorCode
from app.protocols import DocumentService
from app.schemas import SignupRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth_service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth", None)
    if service is None:
        raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, "Authentication is not configured.")
    assert isinstance(service, AuthService)
    return service


AuthDep = Annotated[AuthService, Depends(get_auth_service)]


@router.post(
    "/signup",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account (role is always `user`)",
    responses=problems(409, 422, 429),
)
async def signup(body: SignupRequest, auth: AuthDep) -> UserOut:
    return await auth.signup(body)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="OAuth2 password form → access token (30 min)",
    responses=problems(401, 422, 429),
)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], auth: AuthDep) -> TokenResponse:
    return await auth.login(form.username, form.password)


@router.get("/me", response_model=UserOut, summary="Current user and quota usage", responses=problems(401))
async def me(tenant: TenantDep, auth: AuthDep) -> UserOut:
    return await auth.me(tenant)


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete my account and all its data (§5.8.7)",
    responses=problems(401, 409, 503),
)
async def delete_me(tenant: TenantDep, auth: AuthDep, request: Request) -> None:
    documents = getattr(request.app.state, "documents", None)
    await auth.delete_me(tenant, documents if isinstance(documents, DocumentService) else None)
