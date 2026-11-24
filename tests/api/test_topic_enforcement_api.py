"""Closed Topic list over HTTP: public Documents reject unknown topics with 422."""

from __future__ import annotations

import httpx

from app.core.tenancy import Role
from app.ingestion.metadata import TOPICS
from tests.api import test_ingestion_api as base
from tests.api.test_ingestion_api import DOCS, TEXT, Tenant, files

# Reuse the real-service app fixtures from the ingestion API tests.
api_settings, tenant, app, client = base.api_settings, base.tenant, base.app, base.client


def _assert_topic_problem(r: httpx.Response) -> None:
    assert r.status_code == 422, r.text
    problem = r.json()
    assert problem["code"] == "validation_failed"
    assert problem["extra"]["field"] == "topic"
    assert sorted(problem["extra"]["allowed"]) == sorted(TOPICS)


async def test_admin_public_upload_with_unknown_topic_is_422(
    client: httpx.AsyncClient, tenant: Tenant
) -> None:
    tenant.act_as("u_admin", Role.ADMIN)
    r = await client.post(DOCS, files=files("p.md", TEXT), data={"visibility": "public", "topic": "cnic"})
    _assert_topic_problem(r)
    assert (await client.get(DOCS, params={"scope": "public"})).json()["items"] == []


async def test_admin_public_upload_with_known_topic_is_normalised(
    client: httpx.AsyncClient, tenant: Tenant
) -> None:
    tenant.act_as("u_admin", Role.ADMIN)
    r = await client.post(DOCS, files=files("p.md", TEXT), data={"visibility": "public", "topic": "Passport"})
    assert r.status_code == 201, r.text
    assert r.json()["topic"] == "passport"


async def test_patch_public_document_with_unknown_topic_is_422(
    client: httpx.AsyncClient, tenant: Tenant
) -> None:
    tenant.act_as("u_admin", Role.ADMIN)
    created = await client.post(
        DOCS, files=files("p.md", TEXT), data={"visibility": "public", "topic": "tax"}
    )
    assert created.status_code == 201, created.text
    doc_id = created.json()["id"]
    _assert_topic_problem(await client.patch(f"{DOCS}/{doc_id}", json={"topic": "cnic"}))
    assert (await client.get(f"{DOCS}/{doc_id}")).json()["topic"] == "tax"
