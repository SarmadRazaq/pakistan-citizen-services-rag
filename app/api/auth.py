from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api._stub import not_implemented
from app.api.deps import TenantDep
from app.api.responses import problems
from app.schemas import SignupRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/signup",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account (role is always `user`)",
    responses=problems(409, 422, 429),
)
async def signup(body: SignupRequest) -> UserOut:
    not_implemented("A5")


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="OAuth2 password form → access token (30 min)",
    responses=problems(401, 422, 429),
)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> TokenResponse:
    not_implemented("A5")


@router.get("/me", response_model=UserOut, summary="Current user and quota usage", responses=problems(401))
async def me(tenant: TenantDep) -> UserOut:
    not_implemented("A5")


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete my account and all its data (§5.8.7)",
    responses=problems(401, 409, 503),
)
async def delete_me(tenant: TenantDep) -> None:
    not_implemented("A5")
