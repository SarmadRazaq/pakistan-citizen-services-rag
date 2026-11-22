"""Structural rules from the blueprint, checked on every CI run."""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"
QDRANT_ALLOWED = {APP / "retrieval" / "vector_store.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_qdrant_client_imported_only_in_vector_store() -> None:
    """§5.8.2 / E43: every Qdrant call must go through TenantScopedVectorStore."""
    offenders = [
        str(path.relative_to(APP))
        for path in APP.rglob("*.py")
        if path not in QDRANT_ALLOWED
        and any(name == "qdrant_client" or name.startswith("qdrant_client.") for name in _imports(path))
    ]
    assert offenders == []


def test_routers_do_not_import_storage_or_models_directly() -> None:
    """Routers stay thin: they call services, never Qdrant, the DB layer or model code."""
    forbidden = ("qdrant_client", "sqlalchemy", "app.db", "app.retrieval", "torch")
    offenders = {
        str(path.relative_to(APP)): sorted(n for n in _imports(path) if n.startswith(forbidden))
        for path in (APP / "api").glob("*.py")
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_every_route_documents_its_errors() -> None:
    from app.main import create_app

    app = create_app()
    for route in app.routes:
        path = getattr(route, "path", "")
        if path.startswith("/api/v1"):
            assert getattr(route, "responses", None), f"{path} has no documented error responses"
