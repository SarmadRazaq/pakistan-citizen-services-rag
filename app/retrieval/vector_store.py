"""TenantScopedVectorStore: the only module that talks to Qdrant (blueprint §3.5, §5.3, §5.8.2).

Every public method takes the `TenantContext` and builds its own tenant condition:

- reads: `user_id ∈ ctx.readable_owner_ids` (a `system_context()` reads everything);
- writes: the target owner must satisfy `ctx.can_write`, else `AppError(ADMIN_REQUIRED)`; updates and
  deletes are additionally filtered to the owners the context may write;
- a missing context raises `MissingTenantFilterError` (500 + `security.tenant_filter_missing`, E43).

Before any filter reaches the raw client, `_assert_tenant` checks that it carries the `user_id`
condition, so a future edit that forgets it fails loudly instead of leaking. Every point that comes back
is checked once more against the context (§5.8.2 layer 4).

Collection layout: `chunks__bge-m3__v1` behind the alias `settings.qdrant_collection_alias`, named
vectors `dense` (cosine, HNSW m=16, ef_construct=100) and `sparse` (BGE-M3 lexical weights, no IDF
modifier, §5.8.5).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Awaitable, Iterable, Sequence
from datetime import UTC, date, datetime
from typing import Any, Literal, TypeVar

import httpx
import structlog
from qdrant_client import AsyncQdrantClient, models  # noqa: TID251
from qdrant_client.http.exceptions import (  # noqa: TID251
    ResponseHandlingException,
    UnexpectedResponse,
)

from app.config import Settings
from app.core.errors import AppError, ErrorCode, MissingTenantFilterError
from app.core.tenancy import TenantContext, Visibility
from app.protocols import ChunkPayload, ChunkPoint, Embedding, ScoredPoint, SearchFilters, SparseVector

log = structlog.get_logger(__name__)

T = TypeVar("T")

COLLECTION_NAME = "chunks__bge-m3__v1"
DENSE = "dense"
SPARSE = "sparse"
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
UPSERT_BATCH = 256
SCROLL_PAGE = 1000

KEYWORD_INDEXES = (
    "visibility",
    "doc_id",
    "chunk_hash",
    "agency",
    "jurisdiction",
    "topic",
    "language",
    "file_type",
)
DATETIME_INDEXES = ("effective_date",)

_PAYLOAD_FIELDS = {f.name: f for f in dataclasses.fields(ChunkPayload)}
_UNAVAILABLE = "The vector database is unavailable."
_RESERVED_KEYS = frozenset({"user_id", "visibility", "chunk_id", "created_at"})


# ---------------------------------------------------------------- payload (de)serialisation


def _to_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_to_json_value(v) for v in value]
    return value


def payload_to_qdrant(
    payload: ChunkPayload, *, owner_id: str, visibility: Visibility, point_id: str
) -> dict[str, Any]:
    """ChunkPayload → the §5.3 JSON payload. Dates become ISO strings, `section_path` a list."""
    body = {name: _to_json_value(getattr(payload, name)) for name in _PAYLOAD_FIELDS}
    body["user_id"] = owner_id
    body["visibility"] = visibility.value
    body["chunk_id"] = point_id
    body["created_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return body


def _parse_date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    text = str(value)
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def payload_from_qdrant(raw: dict[str, Any]) -> ChunkPayload:
    """The stored JSON payload → ChunkPayload. Unknown keys (user_id, chunk_id, ...) are ignored."""
    kwargs: dict[str, Any] = {}
    for name in _PAYLOAD_FIELDS:
        if name not in raw:
            continue
        value = raw[name]
        if name == "effective_date":
            value = _parse_date(value)
        elif name == "section_path":
            value = tuple(str(v) for v in (value or ()))
        kwargs[name] = value
    return ChunkPayload(**kwargs)


def payload_patch_to_qdrant(patch: dict[str, Any]) -> dict[str, Any]:
    """Validate a `set_payload` patch: only ChunkPayload fields; owner and scope are never patchable."""
    bad = sorted(k for k in patch if k in _RESERVED_KEYS or k not in _PAYLOAD_FIELDS)
    if bad:
        raise ValueError(f"set_payload cannot change these keys: {bad}")
    return {k: _to_json_value(v) for k, v in patch.items()}


# ---------------------------------------------------------------- vector maths


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0 or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def _dense_similarity(query: Sequence[float], stored: object) -> float:
    """Cosine to the stored dense vector as returned by Qdrant (a flat list of floats)."""
    if not isinstance(stored, list) or not all(isinstance(v, int | float) for v in stored):
        return 0.0
    return cosine(query, [float(v) for v in stored])


def sparse_dot(query: SparseVector, indices: Sequence[int], values: Sequence[float]) -> float:
    weights = dict(zip(indices, values, strict=True))
    return sum(w * weights.get(i, 0.0) for i, w in zip(query.indices, query.values, strict=True))


# ---------------------------------------------------------------- the store


class TenantScopedVectorStore:
    """`VectorStore` over Qdrant. Construct with `from_settings`, or pass an `AsyncQdrantClient`."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        dense_dim: int,
        alias: str = "chunks_active",
        collection: str = COLLECTION_NAME,
    ) -> None:
        self._client = client
        self.dense_dim = dense_dim
        self.alias = alias
        self.collection = collection

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        dense_dim: int,
        location: str | None = None,
        collection: str = COLLECTION_NAME,
    ) -> TenantScopedVectorStore:
        """`location=":memory:"` gives qdrant-client's local in-process mode (tests)."""
        if location is not None:
            client = AsyncQdrantClient(location=location)
        else:
            api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=api_key or None,
                timeout=max(1, math.ceil(settings.qdrant_timeout_s)),
                check_compatibility=False,  # no blocking version probe in the constructor
            )
        return cls(client, dense_dim=dense_dim, alias=settings.qdrant_collection_alias, collection=collection)

    # ------------------------------------------------------------ tenant conditions

    @staticmethod
    def _require(ctx: TenantContext | None, op: str) -> TenantContext:
        if not isinstance(ctx, TenantContext):
            log.error("security.tenant_filter_missing", operation=op)
            raise MissingTenantFilterError(op)
        return ctx

    @staticmethod
    def _owner_condition(owners: Iterable[str]) -> models.FieldCondition:
        return models.FieldCondition(key="user_id", match=models.MatchAny(any=sorted(owners)))

    def _read_conditions(self, ctx: TenantContext, op: str) -> list[models.Condition]:
        if ctx.is_system:
            return []
        if not ctx.readable_owner_ids:
            raise MissingTenantFilterError(op)
        return [self._owner_condition(ctx.readable_owner_ids)]

    @staticmethod
    def _writable_owners(ctx: TenantContext) -> frozenset[str] | None:
        """None means unrestricted (system context only)."""
        if ctx.is_system:
            return None
        owners = {ctx.user_id}
        if ctx.is_admin:
            owners.add(ctx.public_owner_id)
        return frozenset(owners)

    def _write_conditions(self, ctx: TenantContext) -> list[models.Condition]:
        owners = self._writable_owners(ctx)
        return [] if owners is None else [self._owner_condition(owners)]

    @staticmethod
    def _assert_tenant(ctx: TenantContext, flt: models.Filter | None, op: str) -> None:
        """Last line before the raw client: a non-system filter must carry the user_id condition."""
        if ctx.is_system:
            return
        must = flt.must if flt is not None else None
        conditions = must if isinstance(must, list) else [must] if must is not None else []
        if not any(isinstance(c, models.FieldCondition) and c.key == "user_id" for c in conditions):
            log.error("security.tenant_filter_missing", operation=op)
            raise MissingTenantFilterError(op)

    def _check_point_owner(self, ctx: TenantContext, owner_id: str, point_id: str) -> bool:
        if ctx.can_read(owner_id):
            return True
        log.error("security.cross_tenant_point_dropped", point_id=point_id)
        return False

    # ------------------------------------------------------------ error mapping

    @staticmethod
    async def _guard(op: str, call: Awaitable[T]) -> T:
        try:
            return await call
        except AppError:
            raise
        except UnexpectedResponse as exc:
            if exc.status_code is None or exc.status_code >= 500:
                log.error("qdrant.unavailable", operation=op, status=exc.status_code)
                raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, _UNAVAILABLE) from exc
            raise
        except (ResponseHandlingException, httpx.TransportError, ConnectionError, TimeoutError) as exc:
            log.error("qdrant.unavailable", operation=op, exc_type=type(exc).__name__)
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, _UNAVAILABLE) from exc

    # ------------------------------------------------------------ lifecycle

    async def ensure_collection(self) -> None:
        """Create `chunks__bge-m3__v1` (vectors, payload indexes) and point the alias at it, if missing."""
        await self._guard("ensure_collection", self._ensure_collection())

    async def _ensure_collection(self) -> None:
        aliases = await self._client.get_aliases()
        target = next((a.collection_name for a in aliases.aliases if a.alias_name == self.alias), None)
        if target is None:
            if not await self._client.collection_exists(self.collection):
                await self._create_collection()
            await self._client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=self.collection, alias_name=self.alias
                        )
                    )
                ]
            )
            target = self.collection
            log.info("qdrant.alias_created", alias=self.alias, collection=self.collection)
        await self._check_dimension(target)

    async def _create_collection(self) -> None:
        await self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.dense_dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
                )
            },
            sparse_vectors_config={
                # No IDF modifier: BGE-M3 weights already encode term importance, and collection-wide
                # IDF would let other tenants' documents shift a user's scores (§3.5, §5.8.5).
                SPARSE: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False), modifier=models.Modifier.NONE
                )
            },
        )
        await self._client.create_payload_index(
            collection_name=self.collection,
            field_name="user_id",
            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
            wait=True,
        )
        for name in KEYWORD_INDEXES:
            await self._client.create_payload_index(
                collection_name=self.collection,
                field_name=name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        for name in DATETIME_INDEXES:
            await self._client.create_payload_index(
                collection_name=self.collection,
                field_name=name,
                field_schema=models.PayloadSchemaType.DATETIME,
                wait=True,
            )
        log.info("qdrant.collection_created", collection=self.collection, dense_dim=self.dense_dim)

    async def _check_dimension(self, collection: str) -> None:
        info = await self._client.get_collection(collection)
        vectors = info.config.params.vectors
        params = vectors.get(DENSE) if isinstance(vectors, dict) else None
        if params is not None and params.size != self.dense_dim:
            raise RuntimeError(
                f"collection {collection} has {params.size}-d dense vectors, the embedder makes "
                f"{self.dense_dim}-d ones: build a new collection and swap the alias (§3.5)"
            )

    async def ready(self) -> bool:
        try:
            await self._client.get_collection(self.alias)
        except Exception:  # readiness must never raise
            return False
        return True

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------ writes

    async def upsert(self, ctx: TenantContext, owner_id: str, points: Sequence[ChunkPoint]) -> None:
        ctx = self._require(ctx, "upsert")
        if not ctx.can_write(owner_id):
            raise AppError(ErrorCode.ADMIN_REQUIRED, "Only admins can change the official corpus.")
        if not points:
            return
        visibility = Visibility.PUBLIC if owner_id == ctx.public_owner_id else Visibility.PRIVATE
        structs = []
        for p in points:
            if len(p.embedding.dense) != self.dense_dim:
                raise ValueError(f"point {p.point_id}: expected {self.dense_dim}-d dense vector")
            vector: dict[str, Any] = {DENSE: list(p.embedding.dense)}
            if p.embedding.sparse.indices:
                vector[SPARSE] = models.SparseVector(
                    indices=list(p.embedding.sparse.indices), values=list(p.embedding.sparse.values)
                )
            structs.append(
                models.PointStruct(
                    id=p.point_id,
                    vector=vector,
                    payload=payload_to_qdrant(
                        p.payload, owner_id=owner_id, visibility=visibility, point_id=p.point_id
                    ),
                )
            )
        for start in range(0, len(structs), UPSERT_BATCH):
            await self._guard(
                "upsert",
                self._client.upsert(
                    collection_name=self.alias, points=structs[start : start + UPSERT_BATCH], wait=True
                ),
            )

    def _ids_filter(self, ctx: TenantContext, point_ids: Sequence[str], op: str) -> models.Filter:
        flt = models.Filter(
            must=[*self._write_conditions(ctx), models.HasIdCondition(has_id=list(point_ids))]
        )
        self._assert_tenant(ctx, flt, op)
        return flt

    async def set_payload(
        self, ctx: TenantContext, point_ids: Sequence[str], payload: dict[str, Any]
    ) -> None:
        ctx = self._require(ctx, "set_payload")
        body = payload_patch_to_qdrant(payload)
        if not point_ids or not body:
            return
        flt = self._ids_filter(ctx, point_ids, "set_payload")
        await self._guard(
            "set_payload",
            self._client.set_payload(
                collection_name=self.alias, payload=body, points=models.FilterSelector(filter=flt), wait=True
            ),
        )

    async def delete_points(self, ctx: TenantContext, point_ids: Sequence[str]) -> None:
        ctx = self._require(ctx, "delete_points")
        if not point_ids:
            return
        flt = self._ids_filter(ctx, point_ids, "delete_points")
        await self._guard(
            "delete_points",
            self._client.delete(
                collection_name=self.alias, points_selector=models.FilterSelector(filter=flt), wait=True
            ),
        )

    async def delete_document(self, ctx: TenantContext, doc_id: str) -> int:
        """Delete the document's points the context may write (wait=true); return what is left of it.

        The remaining count covers every point of `doc_id` the context can read or write, so a refused
        delete (a non-admin on a public document) shows up as a non-zero result.
        """
        ctx = self._require(ctx, "delete_document")
        doc_cond = models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        flt = models.Filter(must=[*self._write_conditions(ctx), doc_cond])
        self._assert_tenant(ctx, flt, "delete_document")
        await self._guard(
            "delete_document",
            self._client.delete(
                collection_name=self.alias, points_selector=models.FilterSelector(filter=flt), wait=True
            ),
        )
        if ctx.is_system:
            remaining_owner: list[models.Condition] = []
        else:
            writable = self._writable_owners(ctx) or frozenset()
            remaining_owner = [self._owner_condition(ctx.readable_owner_ids | writable)]
        check = models.Filter(must=[*remaining_owner, doc_cond])
        self._assert_tenant(ctx, check, "delete_document")
        result = await self._guard(
            "delete_document",
            self._client.count(collection_name=self.alias, count_filter=check, exact=True),
        )
        if result.count:
            log.warning("qdrant.delete_document_incomplete", doc_id=doc_id, remaining=result.count)
        return int(result.count)

    # ------------------------------------------------------------ reads

    def _read_filter(
        self, ctx: TenantContext, op: str, extra: Sequence[models.Condition] = ()
    ) -> models.Filter | None:
        must = [*self._read_conditions(ctx, op), *extra]
        flt = models.Filter(must=must) if must else None
        self._assert_tenant(ctx, flt, op)
        return flt

    async def point_ids(self, ctx: TenantContext, doc_id: str) -> set[str]:
        ctx = self._require(ctx, "point_ids")
        flt = self._read_filter(
            ctx, "point_ids", [models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        ids: set[str] = set()
        offset: models.ExtendedPointId | None = None
        while True:
            records, offset = await self._guard(
                "point_ids",
                self._client.scroll(
                    collection_name=self.alias,
                    scroll_filter=flt,
                    limit=SCROLL_PAGE,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                ),
            )
            ids.update(str(r.id) for r in records)
            if offset is None:
                return ids

    async def count(self, ctx: TenantContext, doc_id: str | None = None) -> int:
        ctx = self._require(ctx, "count")
        extra = (
            [models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            if doc_id is not None
            else []
        )
        flt = self._read_filter(ctx, "count", extra)
        result = await self._guard(
            "count", self._client.count(collection_name=self.alias, count_filter=flt, exact=True)
        )
        return int(result.count)

    @staticmethod
    def _search_conditions(filters: SearchFilters) -> list[models.Condition] | None:
        """SearchFilters → conditions; None when a filter is an empty tuple (matches nothing)."""
        conditions: list[models.Condition] = []
        for key, values in (
            ("doc_id", filters.doc_ids),
            ("agency", filters.agencies),
            ("jurisdiction", filters.jurisdictions),
        ):
            if values is None:
                continue
            if not values:
                return None
            conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
        return conditions

    async def hybrid_search(
        self,
        ctx: TenantContext,
        query: Embedding,
        *,
        filters: SearchFilters,
        prefetch_k: int,
        limit: int,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    ) -> list[ScoredPoint]:
        """Dense + sparse prefetch, server-side RRF, tenant filter on both legs (§3.6).

        `mode` (eval override) searches one leg only: `dense` or `sparse` query that named vector
        directly with the same tenant filter and return no fused `rrf_score`.

        `similarity` is the dense cosine between the query and *every* returned point (sparse-only hits
        included), computed locally from the returned dense vector. `sparse_score` is the exact sparse
        dot product (no IDF, so it equals Qdrant's sparse score) and `rrf_score` the fused score.
        """
        ctx = self._require(ctx, "hybrid_search")
        conditions = self._search_conditions(filters)
        if conditions is None or limit <= 0:
            return []
        flt = self._read_filter(ctx, "hybrid_search", conditions)
        prefetch_limit = max(1, prefetch_k)  # each leg returns prefetch_k; the fused list is cut to limit
        sparse_query = models.SparseVector(
            indices=list(query.sparse.indices), values=list(query.sparse.values)
        )
        if mode == "hybrid":
            prefetch = [
                models.Prefetch(query=list(query.dense), using=DENSE, filter=flt, limit=prefetch_limit)
            ]
            if query.sparse.indices:
                prefetch.append(
                    models.Prefetch(query=sparse_query, using=SPARSE, filter=flt, limit=prefetch_limit)
                )
            request = self._client.query_points(
                collection_name=self.alias,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=flt,
                with_payload=True,
                with_vectors=[DENSE, SPARSE],
                limit=limit,
            )
        else:
            if mode == "sparse" and not query.sparse.indices:
                return []
            request = self._client.query_points(
                collection_name=self.alias,
                query=list(query.dense) if mode == "dense" else sparse_query,
                using=DENSE if mode == "dense" else SPARSE,
                query_filter=flt,
                with_payload=True,
                with_vectors=[DENSE, SPARSE],
                limit=limit,
            )
        response = await self._guard("hybrid_search", request)
        out: list[ScoredPoint] = []
        for point in response.points:
            raw = point.payload or {}
            owner_id = str(raw.get("user_id", ""))
            pid = str(point.id)
            if not self._check_point_owner(ctx, owner_id, pid):
                continue
            vectors = point.vector if isinstance(point.vector, dict) else {}
            dense_vec = vectors.get(DENSE)
            similarity = _dense_similarity(query.dense, dense_vec)
            sparse_vec = vectors.get(SPARSE)
            sparse_score = (
                sparse_dot(query.sparse, sparse_vec.indices, sparse_vec.values)
                if isinstance(sparse_vec, models.SparseVector)
                else 0.0
            )
            out.append(
                ScoredPoint(
                    point_id=pid,
                    owner_id=owner_id,
                    payload=payload_from_qdrant(raw),
                    similarity=similarity,
                    sparse_score=sparse_score,
                    rrf_score=float(point.score) if mode == "hybrid" else None,
                )
            )
        return out
