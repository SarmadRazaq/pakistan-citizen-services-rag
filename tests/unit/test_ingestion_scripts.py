"""Corpus scripts: seed-list parsing, polite fetching (mock transport, no network), manifest, seeding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.core.tenancy import Role, Scope, TenantContext, system_context
from app.db.session import Database
from app.ingestion.chunking import RegexTokenCounter
from app.ingestion.ocr import NoOcr
from app.ingestion.service import IngestionService
from tests.fakes import FakeEmbedder, InMemoryVectorStore

REPO = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collect = _load("collect_corpus")
seed_mod = _load("seed_corpus")


# ---------------------------------------------------------------- seed list


def test_parse_real_seed_list_selects_script_rows_only() -> None:
    rows = collect.parse_seed_sources((REPO / "corpus" / "SEED_SOURCES.md").read_text(encoding="utf-8"))
    by_id = {r.id: r for r in rows}
    assert {"ID-01", "PP-01", "SP-01", "TX-02", "RC-01", "SD-05"} <= set(by_id)
    targets = collect.plan(rows, root=Path("/repo"))
    ids = {t.id for t in targets}
    assert not any(t.seed.id.startswith("ID-0") and t.seed.id != "ID-07" for t in targets)  # NADRA is manual
    assert "ID-07" in ids  # the archive.org copy is scriptable
    assert not {"SP-01", "SP-02", "SP-03", "SP-04"} & {t.seed.id for t in targets}  # excluded / manual
    assert {"TX-02a", "TX-02b", "TX-02c", "PP-06a", "PP-06b", "PP-06c", "RC-01"} <= ids
    tx02 = [t.url for t in targets if t.seed.id == "TX-02"]
    assert tx02[1] == "https://fbr.gov.pk/categ/active-taxpayer-list-income-tax/51147/30859/71168"
    pp06 = [t for t in targets if t.seed.id == "PP-06"]
    assert [t.path.suffix for t in pp06] == [".jpg", ".png", ".jpg"]
    rc11 = next(t for t in targets if t.id == "RC-11")
    assert rc11.path.parent.as_posix().endswith("corpus/adversarial/raw")  # test rows kept apart
    pp01 = next(t for t in targets if t.id == "PP-01")
    assert (
        pp01.path.name.startswith("PP-01__general-requirements-for-passport") and pp01.path.suffix == ".html"
    )


def test_manual_checklist_urls() -> None:
    urls = collect.parse_manual_urls((REPO / "corpus" / "MANUAL_COLLECTION.md").read_text(encoding="utf-8"))
    assert urls["ID-02b"] == "https://www.nadra.gov.pk/identityDocument/cnic?tab=nic&action=renew"
    assert urls["SP-05"] == "https://www.pmhealthprogram.gov.pk/faqs"


def test_metadata_helpers() -> None:
    assert collect.agency_for("https://download1.fbr.gov.pk/Docs/x.pdf") == "FBR"
    assert collect.agency_for("https://web.archive.org/web/2025/https://www.nadra.gov.pk/fee") == "NADRA"
    assert collect.agency_for("https://sja.gos.pk/assets/x.pdf") == "Sindh"
    assert collect.doc_type_for("Withholding Tax Rate Card TY2027", "PDF") == "fee_schedule"
    assert collect.doc_type_for("Passport Rules 2021", "PDF") == "law"
    assert collect.doc_type_for("PM Health FAQs", "HTML") == "faq"
    assert collect.language_code("**en + ur**") == "en+ur"
    assert collect.slugify("Income Tax Ordinance 2001, amended up to 30-06-2026") == (
        "income-tax-ordinance-2001-amended-up-to-30-06-2026"
    )


# ---------------------------------------------------------------- polite fetching


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def _fetcher(handler: Any) -> tuple[Any, FakeClock, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        response: httpx.Response = handler(request)
        return response

    clock = FakeClock()
    client = httpx.Client(transport=httpx.MockTransport(record), headers={"User-Agent": collect.USER_AGENT})
    return collect.PoliteFetcher(client, clock=clock.time, sleep=clock.sleep), clock, seen


def test_robots_and_per_host_delay() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            if request.url.host == "slow.gov.pk":
                return httpx.Response(200, text="User-agent: *\nCrawl-delay: 5\nDisallow: /private/\n")
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF-1.4 data", headers={"content-type": "application/pdf"})

    fetcher, clock, seen = _fetcher(handler)
    assert fetcher.allowed("https://slow.gov.pk/docs/a.pdf")
    assert not fetcher.allowed("https://slow.gov.pk/private/b.pdf")
    assert fetcher.allowed("https://open.gov.pk/anything")  # 404 robots.txt → allowed
    fetcher.get("https://slow.gov.pk/docs/a.pdf")
    fetcher.get("https://slow.gov.pk/docs/c.pdf")
    fetcher.get("https://open.gov.pk/x")
    fetcher.get("https://open.gov.pk/y")
    assert all(r.headers["user-agent"] == "rag-course-project/1.0" for r in seen)
    # open.gov.pk/x comes 10 s after that host's robots.txt request, so only /y has to wait.
    assert clock.sleeps == [5.0, 5.0, 2.0]  # Crawl-delay honoured; ≥ 2 s per host otherwise


def test_robots_unreachable_disallows() -> None:
    fetcher, _, _ = _fetcher(lambda request: httpx.Response(503))
    assert not fetcher.allowed("https://down.gov.pk/a.pdf")


def test_collect_writes_files_and_manifest(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
        if request.url.path == "/missing.pdf":
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF-1.4 bytes", headers={"content-type": "application/pdf"})

    seed = collect.SeedRow(
        id="PP-03", use="core", title="Passport Rules 2021", urls=(), format="PDF 26 pp", language="en",
        date_shown="Gazette 14 Feb 2022", notes="", collect="script", section="2. Passport",
    )  # fmt: skip
    targets = [
        collect.Target("PP-03", seed, "https://dgip.gov.pk/rules", tmp_path / "corpus/raw/PP-03__rules.html"),
        collect.Target("PP-04", seed, "https://dgip.gov.pk/blocked", tmp_path / "corpus/raw/PP-04__x.pdf"),
        collect.Target(
            "PP-05", seed, "https://dgip.gov.pk/missing.pdf", tmp_path / "corpus/raw/PP-05__y.pdf"
        ),
    ]
    fetcher, _, _ = _fetcher(handler)
    lines: list[str] = []
    manifest = collect.collect(targets, fetcher, {}, root=tmp_path, log=lines.append)
    assert set(manifest) == {"PP-03"}
    row = manifest["PP-03"]
    assert row["file"] == "corpus/raw/PP-03__rules.pdf"  # extension fixed from the Content-Type
    assert (row["agency"], row["jurisdiction"], row["doc_type"]) == ("DGIP", "federal", "law")
    assert row["effective_date"] == "2022-02-14" and row["collected_by"] == "script"
    assert len(row["sha256"]) == 64 and row["derived"] == "no"
    assert any("robots.txt disallows" in line for line in lines)
    assert any("HTTP 404" in line for line in lines)

    path = tmp_path / "corpus/manifest.csv"
    collect.write_manifest(path, manifest)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith(
        "id,url,agency,jurisdiction,doc_type,effective_date,language,sha256,retrieved_at,collected_by,license_note,derived"
    )
    assert collect.read_manifest(path)["PP-03"]["sha256"] == row["sha256"]
    # A second run keeps the existing file and does not download again.
    fetcher2, _, seen2 = _fetcher(handler)
    collect.collect(
        [collect.Target("PP-03", seed, targets[0].url, tmp_path / row["file"])],
        fetcher2,
        manifest,
        root=tmp_path,
    )
    assert seen2 == []


def test_register_manual(tmp_path: Path) -> None:
    rows = collect.parse_seed_sources((REPO / "corpus" / "SEED_SOURCES.md").read_text(encoding="utf-8"))
    manual = tmp_path / "corpus/raw/manual"
    manual.mkdir(parents=True)
    (manual / "ID-02b__nadra-cnic-renew.html").write_text("<html>renew</html>", encoding="utf-8")
    (manual / "ID-05__nadra-registration-policy-6-0-2-en.pdf").write_bytes(b"%PDF-1.4")
    (manual / "SP-01__bisp.html").write_text("excluded", encoding="utf-8")
    (manual / "random.txt").write_text("no id", encoding="utf-8")
    urls = collect.parse_manual_urls((REPO / "corpus" / "MANUAL_COLLECTION.md").read_text(encoding="utf-8"))
    manifest = collect.register_manual(rows, manual, {}, manual_urls=urls, root=tmp_path, log=lambda _: None)
    assert set(manifest) == {"ID-02b", "ID-05"}
    assert manifest["ID-02b"]["url"].endswith("action=renew")
    assert manifest["ID-05"]["collected_by"] == "manual" and manifest["ID-05"]["agency"] == "NADRA"
    assert manifest["ID-05"]["file"] == "corpus/raw/manual/ID-05__nadra-registration-policy-6-0-2-en.pdf"
    dry = collect.register_manual(
        rows, manual, {}, manual_urls=urls, root=tmp_path, dry_run=True, log=lambda _: None
    )
    assert dry == {}


def test_dry_run_makes_no_requests(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_network() -> None:
        raise AssertionError("dry run must not open a client")

    monkeypatch.setattr(collect, "make_client", no_network)
    assert collect.main(["--dry-run", "--only", "PP-01"]) == 0
    assert "1 downloads planned" in capsys.readouterr().out


# ---------------------------------------------------------------- seeding


async def test_seed_is_idempotent(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, app_env="test", storage_dir=tmp_path / "storage", ocr_enabled=False)  # type: ignore[call-arg]
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.create_all()
    store = InMemoryVectorStore()
    service = IngestionService(
        settings, db, store, FakeEmbedder(), token_counter=RegexTokenCounter(), ocr=NoOcr()
    )
    raw = tmp_path / "corpus/raw"
    raw.mkdir(parents=True)
    (raw / "PP-02__faq.md").write_text(
        "# FAQ\n\nHow to renew a passport at the passport office of the directorate general.\n",
        encoding="utf-8",
    )
    (raw / "TX-01__due-dates.md").write_text(
        "# Due dates\n\nIncome tax returns of salaried persons are filed by 30 September every year.\n",
        encoding="utf-8",
    )
    (raw / "PP-05__scan.md").write_text("# Test only\n\nOCR stress test file.\n", encoding="utf-8")
    common = {"jurisdiction": "federal", "doc_type": ""}
    rows = [
        {**common, "id": "PP-02", "use": "core", "file": "corpus/raw/PP-02__faq.md",
         "url": "https://dgip.gov.pk/faq", "agency": "DGIP", "effective_date": "2026-10-23", "title": "FAQ"},
        {**common, "id": "TX-01", "use": "pair", "file": "corpus/raw/TX-01__due-dates.md",
         "url": "https://fbr.gov.pk/due", "agency": "FBR", "effective_date": "", "title": "Due dates"},
        {"id": "PP-05", "use": "test", "file": "corpus/raw/PP-05__scan.md", "url": "https://dgip.gov.pk/sro"},
        {"id": "RC-01", "use": "core", "file": "corpus/raw/missing.pdf", "url": "https://pakistancode.gov.pk/x"},
    ]  # fmt: skip
    try:
        first = await seed_mod.seed(service, db, rows, root=tmp_path, log=lambda _: None)
        assert (first.created, first.skipped, first.failed) == (["PP-02", "TX-01"], ["PP-05", "RC-01"], [])
        user = TenantContext.for_user("u_x", Role.USER, Scope.PUBLIC)
        listed = await service.list(
            user,
            agency=None,
            jurisdiction=None,
            topic=None,
            language=None,
            status=None,
            limit=10,
            cursor=None,
        )
        assert {d.visibility.value for d in listed.items} == {"public"} and len(listed.items) == 2
        faq = next(d for d in listed.items if d.agency == "DGIP")
        assert faq.source_url == "https://dgip.gov.pk/faq" and str(faq.effective_date) == "2026-10-23"
        points = len(store.points)

        again = await seed_mod.seed(service, db, rows, root=tmp_path, log=lambda _: None)
        assert again.created == [] and again.unchanged == ["PP-02", "TX-01"]  # 409 no_change → no-op
        assert len(store.points) == points

        (raw / "PP-02__faq.md").write_text(
            "# FAQ\n\nHow to renew a passport at the passport office of the directorate general.\n\n"
            "New: online renewal.\n",
            encoding="utf-8",
        )
        third = await seed_mod.seed(
            service, db, rows, root=tmp_path, ctx=system_context("seed"), log=lambda _: None
        )
        assert third.replaced == ["PP-02"] and third.unchanged == ["TX-01"]
        assert (await service.get(user, faq.id)).version == 2
    finally:
        await db.dispose()
