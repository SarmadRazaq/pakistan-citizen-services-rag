"""Collect the seed corpus politely and write `corpus/manifest.csv` (blueprint §3.1, decisions D1–D5).

Usage (from the repository root):
    .venv/Scripts/python scripts/collect_corpus.py --dry-run          # print the plan, no network
    .venv/Scripts/python scripts/collect_corpus.py                    # download the `script` rows
    .venv/Scripts/python scripts/collect_corpus.py --only PP-01 TX-05 # a subset
    .venv/Scripts/python scripts/collect_corpus.py --register-manual  # files saved by hand (D2, D3)

What it does:
- parses the tables in `corpus/SEED_SOURCES.md` and keeps rows whose Collect column is `script` and
  whose Use is core / pair / test / core (D4). `excluded` rows (BISP, D1) and `manual` rows (NADRA,
  PM Health, KP Sehat: they block scripts) are never fetched;
- downloads one URL at a time: robots.txt is respected (and its Crawl-delay), at least 2 s between
  requests to the same host, user-agent `rag-course-project/1.0` with no personal details;
- saves to `corpus/raw/<ID>__<slug>.<ext>`, and `test` rows to `corpus/adversarial/raw/`;
- writes `corpus/manifest.csv` (merged with what is already there, one row per file).

`--register-manual` registers the files you saved into `corpus/raw/manual/` (named `<ID>__<slug>.<ext>`,
see `corpus/MANUAL_COLLECTION.md`) with `collected_by=manual`. It never touches the network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.metadata import find_any_date, jurisdiction_for_agency  # noqa: E402

USER_AGENT = "rag-course-project/1.0"
MIN_DELAY_S = 2.0
TIMEOUT_S = 60.0
COLLECT_USES = frozenset({"core", "pair", "test", "core (D4)"})
MANIFEST_FIELDS = (
    "id",
    "url",
    "agency",
    "jurisdiction",
    "doc_type",
    "effective_date",
    "language",
    "sha256",
    "retrieved_at",
    "collected_by",
    "license_note",
    "derived",
    # extra columns used by scripts/seed_corpus.py
    "use",
    "title",
    "file",
)
SEED_FILE = ROOT / "corpus" / "SEED_SOURCES.md"
MANUAL_FILE = ROOT / "corpus" / "MANUAL_COLLECTION.md"
MANIFEST_FILE = ROOT / "corpus" / "manifest.csv"

# Agency by URL host / path fragment (first match wins).
DOMAIN_AGENCY: tuple[tuple[str, str], ...] = (
    ("nadra", "NADRA"),
    ("dgip.gov.pk", "DGIP"),
    ("bisp.gov.pk", "BISP"),
    ("pmhealthprogram", "PM Health"),
    ("sehatcardplus", "KP Sehat"),
    ("fbr.gov.pk", "FBR"),
    ("pakistancode", "MoLJ"),
    ("na.gov.pk", "MoLJ"),
    ("senate.gov.pk", "MoLJ"),
    ("mohtasib.gov.pk", "Mohtasib"),
    ("pmdu.gov.pk", "PMDU"),
    ("federalseed.gov.pk", "PMDU"),
    ("mohr.gov.pk", "MoHR"),
    ("punjab", "Punjab"),
    ("ctplahore", "Punjab"),
    ("sindh", "Sindh"),
    ("sja.gos.pk", "Sindh"),
)
DOC_TYPES: tuple[tuple[str, str], ...] = (
    (r"\bfee|rate card|fee structure|fee chart|fee notices?\b", "fee_schedule"),
    (r"\bfaqs?\b", "faq"),
    (r"\bform\b", "form"),
    (r"\b(act|ordinance|rules|regulations|constitution|sro|gazette)\b", "law"),
    (r"\b(manual|guideline|brochure|how to|steps|procedure|process|charter|policy)\b", "guide"),
)
EXT_BY_CONTENT_TYPE = {
    "application/pdf": "pdf",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "image/jpeg": "jpg",
    "image/png": "png",
}
_URL = re.compile(r"https?://[^\s·`)<>\]]+")
_CONTINUATION = re.compile(r"…(/[^\s·`)<>\]]+)")
_MANUAL_NAME = re.compile(r"^(?P<id>[A-Z]{2}-\d{2}[a-z]?)__(?P<slug>[^.]+)\.(?P<ext>[A-Za-z0-9]+)$")


@dataclass(frozen=True, slots=True)
class SeedRow:
    id: str
    use: str
    title: str
    urls: tuple[str, ...]
    format: str
    language: str
    date_shown: str
    notes: str
    collect: str
    section: str


@dataclass(frozen=True, slots=True)
class Target:
    id: str  # seed ID, with a/b/c when one row lists several URLs
    seed: SeedRow
    url: str
    path: Path


# ---------------------------------------------------------------- parsing


def _cells(line: str) -> list[str]:
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") else inner
    return [c.strip() for c in re.split(r"(?<!\\)\|", inner)]


def _plain(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).strip()


def extract_urls(cell: str) -> tuple[str, ...]:
    """Every URL in a cell; `…/71168` continues the previous URL with a new last path segment."""
    urls: list[str] = []
    for token in re.split(r"\s+·\s+|\s+", cell):
        if m := _URL.match(token):
            urls.append(m.group(0).rstrip(".,;"))
        elif (c := _CONTINUATION.match(token)) and urls:
            urls.append(urls[-1].rsplit("/", 1)[0] + c.group(1).rstrip(".,;"))
    return tuple(dict.fromkeys(urls))


def parse_seed_sources(text: str) -> list[SeedRow]:
    rows: list[SeedRow] = []
    section = ""
    header: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            section, header = line[3:].strip(), None
            continue
        if not line.lstrip().startswith("|"):
            header = None
            continue
        cells = _cells(line)
        if header is None:
            if "ID" in cells and "URL" in cells:
                header = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        record = dict(zip(header, cells, strict=False))
        if not re.fullmatch(r"[A-Z]{2}-\d{2}", record.get("ID", "")):
            continue
        rows.append(
            SeedRow(
                id=record["ID"],
                use=_plain(record.get("Use", "")),
                title=_plain(record.get("Title", "")),
                urls=extract_urls(record.get("URL", "")),
                format=_plain(record.get("Format", "")),
                language=_plain(record.get("Lang", "")),
                date_shown=_plain(record.get("Date shown", "")),
                notes=record.get("Notes", ""),
                collect=_plain(record.get("Collect", "")),
                section=section,
            )
        )
    return rows


def parse_manual_urls(text: str) -> dict[str, str]:
    """`ID → URL` from the checklist tables in MANUAL_COLLECTION.md (e.g. ID-02a, ID-02b)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _cells(line)
        if len(cells) < 3 or not re.fullmatch(r"[A-Z]{2}-\d{2}[a-z]?", cells[1]):
            continue
        urls = extract_urls(cells[2])
        if urls:
            out[cells[1]] = urls[0]
    return out


def wanted(row: SeedRow) -> bool:
    return row.collect.lower().startswith("script") and row.use in COLLECT_USES


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "document"


def guess_ext(url: str, fmt: str) -> str:
    suffix = Path(urlsplit(url).path).suffix.lower().lstrip(".")
    if suffix in {"pdf", "jpg", "jpeg", "png", "html", "htm"}:
        return {"jpeg": "jpg", "htm": "html"}.get(suffix, suffix)
    return "pdf" if fmt.upper().startswith("PDF") else "html"


def plan(rows: Iterable[SeedRow], root: Path = ROOT, only: Sequence[str] = ()) -> list[Target]:
    targets: list[Target] = []
    for row in rows:
        if not wanted(row) or (only and row.id not in only):
            continue
        folder = root / "corpus" / ("adversarial/raw" if row.use == "test" else "raw")
        multi = len(row.urls) > 1
        for i, url in enumerate(row.urls):
            target_id = f"{row.id}{chr(ord('a') + i)}" if multi else row.id
            slug = slugify(row.title)
            targets.append(
                Target(target_id, row, url, folder / f"{target_id}__{slug}.{guess_ext(url, row.format)}")
            )
    return targets


# ---------------------------------------------------------------- manifest


def agency_for(url: str) -> str | None:
    lowered = url.lower()
    return next((agency for fragment, agency in DOMAIN_AGENCY if fragment in lowered), None)


def doc_type_for(title: str, fmt: str) -> str:
    for pattern, doc_type in DOC_TYPES:
        if re.search(pattern, title, re.IGNORECASE):
            return doc_type
    return "web_page" if fmt.upper().startswith("HTML") else "document"


def license_for(url: str, agency: str | None) -> str:
    if "web.archive.org" in url:
        return "third-party archive copy (web.archive.org) of an official page; not committed"
    if agency == "FBR":
        return "FBR Terms of Use: unmodified copies for free non-commercial information sharing"
    return "official public document; non-commercial coursework use; no redistribution licence; not committed"


def language_code(cell: str) -> str:
    codes = re.findall(r"\b(en|ur)\b", cell.lower())
    return "+".join(dict.fromkeys(codes)) or ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def manifest_row(
    target_id: str,
    seed: SeedRow,
    url: str,
    path: Path,
    *,
    collected_by: str,
    retrieved_at: str,
    root: Path = ROOT,
) -> dict[str, str]:
    agency = agency_for(url)
    effective = find_any_date(seed.date_shown)
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    return {
        "id": target_id,
        "url": url,
        "agency": agency or "",
        "jurisdiction": jurisdiction_for_agency(agency) or "",
        "doc_type": doc_type_for(seed.title, seed.format),
        "effective_date": effective.isoformat() if effective else "",
        "language": language_code(seed.language),
        "sha256": sha256_of(path),
        "retrieved_at": retrieved_at,
        "collected_by": collected_by,
        "license_note": license_for(url, agency),
        "derived": "no",
        "use": seed.use,
        "title": seed.title,
        "file": rel,
    }


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def write_manifest(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(MANIFEST_FIELDS), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({field: rows[key].get(field, "") for field in MANIFEST_FIELDS})


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------- polite fetching


class PoliteFetcher:
    """One request at a time; robots.txt and a per-host delay of at least `min_delay` seconds."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        min_delay: float = MIN_DELAY_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.min_delay = min_delay
        self.clock = clock
        self.sleep = sleep
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last: dict[str, float] = {}

    @staticmethod
    def _origin(url: str) -> str:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}"

    def _wait(self, origin: str, delay: float) -> None:
        last = self._last.get(origin)
        if last is not None:
            remaining = delay - (self.clock() - last)
            if remaining > 0:
                self.sleep(remaining)
        self._last[origin] = self.clock()

    def robots(self, url: str) -> RobotFileParser | None:
        """Parsed robots.txt; None means "allow everything" (404 and other 4xx answers, RFC 9309)."""
        origin = self._origin(url)
        if origin not in self._robots:
            parser = RobotFileParser()
            self._wait(origin, self.min_delay)
            try:
                response = self.client.get(f"{origin}/robots.txt", follow_redirects=True)
            except httpx.HTTPError:
                parser.disallow_all = True  # unreachable: assume nothing may be fetched
                self._robots[origin] = parser
                return parser
            if response.status_code >= 500:
                parser.disallow_all = True
            elif response.status_code >= 400:
                self._robots[origin] = None
                return None
            else:
                parser.parse(response.text.splitlines())
            self._robots[origin] = parser
        return self._robots[origin]

    def delay_for(self, url: str) -> float:
        parser = self.robots(url)
        crawl_delay = parser.crawl_delay(USER_AGENT) if parser is not None else None
        return max(self.min_delay, float(crawl_delay or 0))

    def allowed(self, url: str) -> bool:
        parser = self.robots(url)
        return True if parser is None else parser.can_fetch(USER_AGENT, url)

    def get(self, url: str) -> httpx.Response:
        self._wait(self._origin(url), self.delay_for(url))
        return self.client.get(url, follow_redirects=True)


def make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)


def collect(
    targets: Sequence[Target],
    fetcher: PoliteFetcher,
    manifest: dict[str, dict[str, str]],
    *,
    force: bool = False,
    root: Path = ROOT,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, str]]:
    for target in targets:
        path = target.path
        if path.exists() and not force:
            retrieved = manifest.get(target.id, {}).get("retrieved_at") or _now()
            manifest[target.id] = manifest_row(
                target.id,
                target.seed,
                target.url,
                path,
                collected_by="script",
                retrieved_at=retrieved,
                root=root,
            )
            log(f"keep   {target.id}: {path.name} (already downloaded)")
            continue
        if not fetcher.allowed(target.url):
            log(f"skip   {target.id}: robots.txt disallows {target.url}")
            continue
        try:
            response = fetcher.get(target.url)
        except httpx.HTTPError as exc:
            log(f"error  {target.id}: {type(exc).__name__} for {target.url}")
            continue
        if response.status_code != 200:
            log(f"error  {target.id}: HTTP {response.status_code} for {target.url}")
            continue
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        ext = EXT_BY_CONTENT_TYPE.get(content_type)
        if ext and path.suffix.lstrip(".") != ext:
            path = path.with_suffix(f".{ext}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        manifest[target.id] = manifest_row(
            target.id, target.seed, target.url, path, collected_by="script", retrieved_at=_now(), root=root
        )
        log(f"saved  {target.id}: {path.name} ({len(response.content):,} bytes)")
    return manifest


def register_manual(
    rows: Sequence[SeedRow],
    manual_dir: Path,
    manifest: dict[str, dict[str, str]],
    *,
    manual_urls: dict[str, str] | None = None,
    root: Path = ROOT,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, str]]:
    by_id = {r.id: r for r in rows}
    manual_urls = manual_urls or {}
    if not manual_dir.exists():
        log(f"nothing to register: {manual_dir} does not exist")
        return manifest
    for path in sorted(p for p in manual_dir.iterdir() if p.is_file()):
        m = _MANUAL_NAME.match(path.name)
        if not m:
            log(f"skip   {path.name}: name is not <ID>__<slug>.<ext>")
            continue
        file_id = m.group("id")
        seed = by_id.get(file_id[:5])
        if seed is None:
            log(f"skip   {path.name}: {file_id} is not in SEED_SOURCES.md")
            continue
        if seed.use.startswith("excluded"):
            log(f"skip   {path.name}: {seed.id} is excluded ({seed.use})")
            continue
        url = manual_urls.get(file_id) or (seed.urls[0] if seed.urls else "")
        saved = datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0).isoformat()
        if dry_run:
            log(f"would register {file_id}: {path.name} ← {url}")
            continue
        manifest[file_id] = manifest_row(
            file_id, seed, url, path, collected_by="manual", retrieved_at=saved, root=root
        )
        log(f"registered {file_id}: {path.name}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=Path, default=SEED_FILE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    parser.add_argument("--dry-run", action="store_true", help="only print the plan; no network, no writes")
    parser.add_argument("--register-manual", action="store_true", help="register files in corpus/raw/manual/")
    parser.add_argument(
        "--only", nargs="*", default=(), help="seed IDs to collect (default: all script rows)"
    )
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args(argv)

    rows = parse_seed_sources(args.seed.read_text(encoding="utf-8"))
    manifest = read_manifest(args.manifest)

    if args.register_manual:
        manual_urls = (
            parse_manual_urls(MANUAL_FILE.read_text(encoding="utf-8")) if MANUAL_FILE.exists() else {}
        )
        manifest = register_manual(
            rows, ROOT / "corpus" / "raw" / "manual", manifest, manual_urls=manual_urls, dry_run=args.dry_run
        )
        if not args.dry_run:
            write_manifest(args.manifest, manifest)
        return 0

    targets = plan(rows, only=args.only)
    if args.dry_run:
        for t in targets:
            print(f"{t.id:7} {t.seed.use:10} {t.url}\n        → {t.path.relative_to(ROOT).as_posix()}")
        skipped = [r for r in rows if not wanted(r)]
        print(f"\n{len(targets)} downloads planned; {len(skipped)} rows not collected by script:")
        for r in skipped:
            print(f"  {r.id:6} use={r.use!r} collect={r.collect!r}")
        return 0

    with make_client() as client:
        manifest = collect(targets, PoliteFetcher(client), manifest)
    write_manifest(args.manifest, manifest)
    print(f"manifest: {args.manifest} ({len(manifest)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
