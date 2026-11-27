from __future__ import annotations

from typing import NoReturn

from app.core.errors import AppError, ErrorCode


def not_implemented(work_package: str) -> NoReturn:
    """Contract stub: the route and its schema are frozen; the named work package fills it in."""
    raise AppError(ErrorCode.NOT_IMPLEMENTED, f"Implemented in work package {work_package}.")
