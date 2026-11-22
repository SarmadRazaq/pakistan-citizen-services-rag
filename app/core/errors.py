"""Domain errors and their RFC 9457 `application/problem+json` mapping (blueprint §8.3).

Services raise `AppError(ErrorCode.X, ...)`. They never build HTTP responses themselves.
The handlers registered by `install_error_handlers` turn every failure into the same envelope:

    {"type", "title", "status", "code", "detail", "request_id", "extra", "errors"}

The `ErrorCode` enum is part of the frozen contract. Add new codes; never rename existing ones.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.context import get_request_id

PROBLEM_JSON = "application/problem+json"
ERROR_TYPE_BASE = "https://rag.example/errors/"

log = structlog.get_logger(__name__)


class ErrorCode(Enum):
    """Machine-readable error codes. Value = (HTTP status, human title)."""

    # 400: bad input or unsafe content
    EMPTY_FILE = (400, "Empty file")
    NO_EXTRACTABLE_TEXT = (400, "No extractable text")
    CORRUPT_FILE = (400, "Corrupt or unreadable file")
    ENCRYPTED_DOCUMENT = (400, "Encrypted document")
    MALICIOUS_ARCHIVE = (400, "Malicious archive")
    MALICIOUS_IMAGE = (400, "Malicious or oversized image")
    ACTIVE_CONTENT_REJECTED = (400, "Active content rejected")
    MALWARE_DETECTED = (400, "Malware detected")
    OBFUSCATED_INPUT = (400, "Obfuscated input")
    PROMPT_INJECTION_DETECTED = (400, "Prompt injection detected")
    JAILBREAK_DETECTED = (400, "Jailbreak attempt detected")
    BAD_REQUEST = (400, "Bad request")
    # 401 / 403 / 404
    UNAUTHORIZED = (401, "Authentication required")
    INVALID_CREDENTIALS = (401, "Invalid credentials")
    ADMIN_REQUIRED = (403, "Admin role required")
    NOT_FOUND = (404, "Not found")
    # 409: conflicts
    EMAIL_TAKEN = (409, "Email already registered")
    DUPLICATE_DOCUMENT = (409, "Duplicate document")
    DUPLICATE_CONTENT = (409, "Duplicate content")
    DUPLICATE_PUBLIC_DOCUMENT = (409, "Document already in the official corpus")
    NO_CHANGE = (409, "No change")
    LAST_ADMIN = (409, "Cannot delete the last admin")
    # 413 / 415 / 422 / 429
    FILE_TOO_LARGE = (413, "File too large")
    QUOTA_EXCEEDED = (413, "Quota exceeded")
    UNSUPPORTED_MEDIA_TYPE = (415, "Unsupported media type")
    QUESTION_EMPTY = (422, "Question is empty")
    QUESTION_TOO_LONG = (422, "Question too long")
    VALIDATION_FAILED = (422, "Request validation failed")
    RATE_LIMITED = (429, "Too many requests")
    TOO_MANY_INGEST_JOBS = (429, "Too many ingestion jobs")
    # 5xx
    INTERNAL_ERROR = (500, "Internal server error")
    TENANT_FILTER_MISSING = (500, "Internal server error")  # security event; same public title as 500
    NOT_IMPLEMENTED = (501, "Not implemented yet")
    LLM_UNAVAILABLE = (503, "Answer model unavailable")
    DEPENDENCY_UNAVAILABLE = (503, "Dependency unavailable")

    @property
    def status(self) -> int:
        return int(self.value[0])

    @property
    def title(self) -> str:
        return str(self.value[1])

    @property
    def slug(self) -> str:
        return self.name.lower()


class AppError(Exception):
    """A domain failure with a stable code. Raised by services, mapped once in this module."""

    def __init__(
        self,
        code: ErrorCode,
        detail: str | None = None,
        *,
        extra: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail or code.title)
        self.code = code
        self.detail = detail or code.title
        self.extra = extra or {}
        self.headers = headers or {}


class MissingTenantFilterError(AppError):
    """Raised when a vector-store call reaches Qdrant without a tenant condition (§5.8.2, E43)."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            ErrorCode.TENANT_FILTER_MISSING,
            "An internal error occurred.",
            extra={},
        )
        self.operation = operation


def problem_body(
    code: ErrorCode,
    detail: str,
    *,
    extra: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "type": ERROR_TYPE_BASE + code.slug,
        "title": code.title,
        "status": code.status,
        "code": code.slug,
        "detail": detail,
        "request_id": get_request_id(),
        "extra": extra or {},
        "errors": errors or [],
    }


def problem_response(
    code: ErrorCode,
    detail: str,
    *,
    extra: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=code.status,
        content=problem_body(code, detail, extra=extra, errors=errors),
        media_type=PROBLEM_JSON,
        headers=headers,
    )


_HTTP_STATUS_TO_CODE: dict[int, ErrorCode] = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.ADMIN_REQUIRED,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.BAD_REQUEST,
    413: ErrorCode.FILE_TOO_LARGE,
    415: ErrorCode.UNSUPPORTED_MEDIA_TYPE,
    429: ErrorCode.RATE_LIMITED,
    501: ErrorCode.NOT_IMPLEMENTED,
    503: ErrorCode.DEPENDENCY_UNAVAILABLE,
}


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    if exc.code is ErrorCode.TENANT_FILTER_MISSING:
        log.error("security.tenant_filter_missing", operation=getattr(exc, "operation", None))
    elif exc.code.status >= 500:
        log.error("app_error", code=exc.code.slug, detail=exc.detail)
    return problem_response(exc.code, exc.detail, extra=exc.extra, headers=exc.headers)


async def _validation_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {
            "loc": [str(part) for part in err.get("loc", ())],
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        for err in exc.errors()
    ]
    return problem_response(ErrorCode.VALIDATION_FAILED, "One or more fields are invalid.", errors=errors)


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _HTTP_STATUS_TO_CODE.get(exc.status_code, ErrorCode.BAD_REQUEST)
    detail = exc.detail if isinstance(exc.detail, str) else code.title
    # Keep the real status (e.g. 405) even when the code is a generic one.
    response = problem_response(code, detail, headers=getattr(exc, "headers", None))
    response.status_code = exc.status_code
    return response


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    # The stack trace goes to the log only, never to the client.
    log.exception("unhandled_exception", exc_type=type(exc).__name__)
    return problem_response(ErrorCode.INTERNAL_ERROR, "An internal error occurred.")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
