"""Password hashing and JWT access tokens (blueprint §12.6, E42).

- Passwords: argon2id through `pwdlib`. Plain passwords are never stored or logged.
- Tokens: HS256 with `sub`, `jti`, `ver`, `iat` and `exp`. `ver` must match `users.token_version`
  (checked by `AuthService.resolve`), so bumping the version kills every token at once.
  Only HS256 is accepted on decode, so `alg=none` and algorithm-confusion tokens are rejected.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.config import Settings
from app.core.errors import AppError, ErrorCode

JWT_ALGORITHM = "HS256"
DEFAULT_JWT_SECRET = "change-me-in-.env"  # noqa: S105 (the placeholder we refuse in prod, not a secret)
REQUIRED_CLAIMS = ("sub", "jti", "ver", "iat", "exp")
BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}

_password_hash = PasswordHash((Argon2Hasher(),))
# Verified when the email is unknown, so a miss costs the same argon2 work as a wrong password.
_DUMMY_HASH = _password_hash.hash("timing-equaliser-not-a-real-password")


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str | None) -> tuple[bool, str | None]:
    """Return (valid, new_hash). `new_hash` is set when the stored hash should be upgraded.

    A missing hash still runs one argon2 verification against a dummy hash (timing equaliser).
    """
    if password_hash is None:
        _password_hash.verify(password, _DUMMY_HASH)
        return False, None
    try:
        return _password_hash.verify_and_update(password, password_hash)
    except Exception:  # a malformed stored hash is a failed login, never a 500
        return False, None


def ensure_secure_secret(settings: Settings) -> None:
    """Refuse to run in production with the placeholder JWT secret."""
    if settings.app_env == "prod" and settings.jwt_secret.get_secret_value() in {"", DEFAULT_JWT_SECRET}:
        raise RuntimeError("JWT_SECRET is still the default. Set a long random JWT_SECRET in .env.")


@dataclass(frozen=True, slots=True)
class TokenClaims:
    sub: str
    jti: str
    ver: int
    iat: int
    exp: int


def _unauthorized(detail: str) -> AppError:
    return AppError(ErrorCode.UNAUTHORIZED, detail, headers=dict(BEARER_CHALLENGE))


def create_access_token(
    settings: Settings, user_id: str, token_version: int, *, now: datetime | None = None
) -> tuple[str, int]:
    """Return (token, expires_in_seconds)."""
    issued = now or datetime.now(UTC)
    ttl = timedelta(minutes=settings.jwt_ttl_min)
    claims: dict[str, Any] = {
        "sub": user_id,
        "jti": uuid.uuid4().hex,
        "ver": token_version,
        "iat": int(issued.timestamp()),
        "exp": int((issued + ttl).timestamp()),
    }
    token = jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=JWT_ALGORITHM)
    return token, int(ttl.total_seconds())


def decode_access_token(settings: Settings, token: str) -> TokenClaims:
    """Verify signature, algorithm and expiry. Any failure is a 401 `unauthorized`."""
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[JWT_ALGORITHM],
            options={"require": list(REQUIRED_CLAIMS)},
        )
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("The access token has expired. Sign in again.") from exc
    except jwt.PyJWTError as exc:
        raise _unauthorized("The access token is invalid.") from exc
    sub, ver = claims.get("sub"), claims.get("ver")
    if not isinstance(sub, str) or not sub or not isinstance(ver, int) or isinstance(ver, bool):
        raise _unauthorized("The access token is invalid.")
    return TokenClaims(
        sub=sub, jti=str(claims["jti"]), ver=ver, iat=int(claims["iat"]), exp=int(claims["exp"])
    )
