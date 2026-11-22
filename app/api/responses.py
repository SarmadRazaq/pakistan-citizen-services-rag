"""OpenAPI `responses=` helpers so every route documents its error statuses with the Problem schema."""

from __future__ import annotations

from typing import Any

from app.schemas import Problem

_DESCRIPTIONS = {
    400: "Bad request or unsafe content",
    401: "Missing or invalid token",
    403: "Admin role required",
    404: "Not found (also returned for another user's private resource)",
    409: "Conflict (duplicate, no change)",
    413: "File too large or quota exceeded",
    415: "Unsupported or spoofed file type",
    422: "Validation failed",
    429: "Rate limit exceeded",
    501: "Not implemented yet",
    503: "A dependency (LLM, Qdrant) is unavailable",
}


def problems(*statuses: int) -> dict[int | str, dict[str, Any]]:
    return {
        status: {
            "model": Problem,
            "description": _DESCRIPTIONS.get(status, "Error"),
            "content": {"application/problem+json": {}},
        }
        for status in statuses
    }
