"""Ingest the official corpus (manifest core/pair rows) into the public scope (blueprint §5.4, §5.8.1).

Usage (from the repository root, after `collect_corpus.py`):
    .venv/Scripts/python scripts/seed_corpus.py [--manifest corpus/manifest.csv] [--uses core pair]
        [--components app.retrieval:build_ingest_components]

Every row goes through the normal ingestion path (`IngestionService`) with `system_context("seed")`,
owned by `u_public`. Re-running is idempotent:
- a row whose source URL is already in the public corpus is sent as a `PUT` (diff upsert): identical
  bytes answer 409 `no_change` → "unchanged"; a changed file becomes version N+1, re-embedding only
  the changed chunks;
- an exact duplicate (409 `duplicate_document` / `duplicate_content`) is also "unchanged".

`--components MODULE:FUNC` names a function `(settings) -> (VectorStore, Embedder)`. The retrieval
work package provides the real BGE-M3 embedder and Qdrant store; the default name is the integration
point the main session wires up.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.core.errors import AppError, ErrorCode  # noqa: E402
from app.core.tenancy import TenantContext, system_context  # noqa: E402
from app.db.models import Document  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.ingestion.service import IngestionService  # noqa: E402
from app.protocols import UploadedFile  # noqa: E402
from app.schemas import DocumentMetadata, DocumentStatus, Jurisdiction  # noqa: E402

MANIFEST_FILE = ROOT / "corpus" / "manifest.csv"
DEFAULT_USES = ("core", "pair", "core (D4)")
DEFAULT_COMPONENTS = "app.retrieval:build_ingest_components"
_UNCHANGED = {ErrorCode.NO_CHANGE, ErrorCode.DUPLICATE_DOCUMENT, ErrorCode.DUPLICATE_CONTENT}


@dataclass
class SeedReport:
    created: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"created={len(self.created)} replaced={len(self.replaced)} unchanged={len(self.unchanged)} "
            f"skipped={len(self.skipped)} failed={len(self.failed)}"
        )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metadata(row: dict[str, str]) -> DocumentMetadata:
    jurisdiction = row.get("jurisdiction") or None
    effective = row.get("effective_date") or None
    return DocumentMetadata(
        title=(row.get("title") or None),
        agency=(row.get("agency") or None),
        jurisdiction=Jurisdiction(jurisdiction) if jurisdiction in {j.value for j in Jurisdiction} else None,
        effective_date=date.fromisoformat(effective) if effective else None,
    )


async def _existing_public_doc(db: Database, ctx: TenantContext, source_url: str) -> str | None:
    async with db.session(ctx) as s:
        row = (
            await s.scalars(
                select(Document.id).where(
                    Document.user_id == ctx.public_owner_id,
                    Document.source_url == source_url,
                    # a FAILED row is retried as a fresh upload (which purges it), not replaced:
                    # replacing it with the same bytes would answer no_change and never retry
                    Document.status != DocumentStatus.FAILED.value,
                )
            )
        ).first()
    return row


async def seed(
    service: IngestionService,
    db: Database,
    rows: Iterable[dict[str, str]],
    *,
    root: Path = ROOT,
    uses: Iterable[str] = DEFAULT_USES,
    ctx: TenantContext | None = None,
    log: Callable[[str], None] = print,
) -> SeedReport:
    ctx = ctx or system_context("seed")
    allowed = set(uses)
    report = SeedReport()
    for row in rows:
        row_id = row.get("id", "?")
        if row.get("use") not in allowed:
            report.skipped.append(row_id)
            continue
        path = root / row.get("file", "")
        if not row.get("file") or not path.is_file():
            log(f"skip    {row_id}: file missing ({row.get('file')})")
            report.skipped.append(row_id)
            continue
        upload = UploadedFile(file_name=path.name, path=path, size_bytes=path.stat().st_size)
        url = row.get("url") or None
        try:
            existing = await _existing_public_doc(db, ctx, url) if url else None
            if existing:
                doc, accepted = await service.replace(ctx, existing, upload)
            else:
                doc, accepted = await service.upload(
                    ctx,
                    upload,
                    visibility="public",
                    metadata=_metadata(row),
                    source_url=url,
                    doc_type=row.get("doc_type") or None,
                )
            if accepted:  # large file: ingested by a background job; wait for it before going on
                pages = doc.progress.pages_total if doc.progress else "?"
                log(f"wait    {row_id}: background ingestion ({pages} pages)")
                await service.drain()
                doc = await service.get(ctx, doc.id)
            if doc.status is DocumentStatus.FAILED:
                report.failed.append(row_id)
                log(f"failed  {row_id}: {doc.error}")
            elif existing:
                report.replaced.append(row_id)
                log(f"updated {row_id}: v{doc.version} ({doc.last_ingest})")
            else:
                report.created.append(row_id)
                log(f"added   {row_id}: {doc.chunk_count} chunks")
        except AppError as exc:
            if exc.code in _UNCHANGED:
                report.unchanged.append(row_id)
                log(f"same    {row_id}: {exc.code.slug}")
            elif exc.code is ErrorCode.UNSUPPORTED_MEDIA_TYPE:
                report.skipped.append(row_id)
                log(f"skip    {row_id}: {exc.detail}")
            else:
                report.failed.append(row_id)
                log(f"failed  {row_id}: {exc.code.slug}: {exc.detail}")
    return report


def load_components(spec: str, settings: Settings) -> tuple[Any, Any]:
    module_name, _, func_name = spec.partition(":")
    try:
        factory = getattr(importlib.import_module(module_name), func_name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(
            f"Cannot load {spec!r} ({exc}). It must be a function (settings) -> (VectorStore, Embedder), "
            "provided by the retrieval work package."
        ) from exc
    store, embedder = factory(settings)
    return store, embedder


async def amain(args: argparse.Namespace) -> int:
    settings = get_settings()
    db = Database(settings.database_url)
    await db.create_all()
    store, embedder = load_components(args.components, settings)
    ensure = getattr(store, "ensure_collection", None)
    if ensure is not None:
        await ensure()
    service = IngestionService(settings, db, store, embedder)
    try:
        report = await seed(service, db, read_manifest(args.manifest), uses=args.uses)
    finally:
        await db.dispose()
    print(report.summary())
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    parser.add_argument("--uses", nargs="*", default=list(DEFAULT_USES))
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    return asyncio.run(amain(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
