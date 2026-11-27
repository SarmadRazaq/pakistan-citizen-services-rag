"""HTML extraction: trafilatura main content + selectolax tables (blueprint §3.2).

1. selectolax removes `<script>`, `<style>` and other active or invisible elements (scripts are
   stripped, never executed: GD-7) and hidden elements (`display:none`, `hidden`; GD-9, counted).
2. Every leaf `<table>` is converted to Markdown with selectolax and replaced by a placeholder
   paragraph, so trafilatura's boilerplate removal cannot mangle it and its position is kept.
3. Expanded accordion questions (`<button aria-expanded="true">`, as on the NADRA FAQ pages)
   become headings; trafilatura drops buttons otherwise and the answer would lose its question.
4. trafilatura's XML output gives headings (→ `section_path`), paragraphs, lists and the placeholders
   in reading order. Repeated blocks (responsive duplicates, repeated tab panes) are dropped.
"""

from __future__ import annotations

import html as html_lib
import re
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as SafeET  # type: ignore[import-untyped]
import trafilatura
from selectolax.lexbor import LexborHTMLParser, LexborNode

from app.core.errors import AppError, ErrorCode
from app.ingestion.extractors.base import SectionTracker, decode_text, dedupe_blocks, rows_to_markdown
from app.protocols import ExtractedBlock, ExtractedDocument

_REMOVE = "script, style, noscript, template, iframe, object, embed, svg, canvas, button, nav, footer"
_HIDDEN = (
    '[hidden], [aria-hidden="true"], [style*="display:none"], [style*="display: none"], '
    '[style*="visibility:hidden"], [style*="visibility: hidden"]'
)
_SAVED_FROM = re.compile(r"<!--\s*saved from url=\(\d+\)(\S+?)\s*-->", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^RAGTABLE(\d+)PLACEHOLDER$")
_HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}


def _node_text(node: LexborNode) -> str:
    return " ".join((node.text(separator=" ", strip=True) or "").split())


def _table_to_markdown(table: LexborNode) -> str:
    rows: list[list[str]] = []
    for tr in table.css("tr"):
        cells = [_node_text(c) for c in tr.css("th, td")]
        if any(cells):
            rows.append(cells)
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return ""
    return rows_to_markdown(rows)


def _source_url(raw: str, tree: LexborHTMLParser) -> str | None:
    if m := _SAVED_FROM.search(raw[:2000]):  # Chrome/Edge "Webpage, Complete" marker
        return m.group(1)
    for selector, attr in (('link[rel="canonical"]', "href"), ('meta[property="og:url"]', "content")):
        node = tree.css_first(selector)
        if node is not None and (value := node.attributes.get(attr)):
            return value
    return None


def _title(tree: LexborHTMLParser) -> str | None:
    for selector in ('meta[property="og:title"]', "title", "h1"):
        node = tree.css_first(selector)
        if node is None:
            continue
        value = node.attributes.get("content") if node.tag == "meta" else _node_text(node)
        if value and value.strip():
            return " ".join(value.split())[:300]
    return None


def _xml_text(el: Any) -> str:
    return " ".join("".join(el.itertext()).split())


class HtmlExtractor:
    file_types = frozenset({"html"})

    def extract(self, path: Path) -> ExtractedDocument:
        raw = decode_text(path.read_bytes(), "HTML file")
        tree = LexborHTMLParser(raw)
        source_url = _source_url(raw, tree)
        title = _title(tree)

        hidden = 0
        for node in tree.css(_HIDDEN):
            if _node_text(node):
                hidden += 1
            node.decompose()
        for button in tree.css('button[aria-expanded="true"]'):  # open accordion → keep its question
            text = _node_text(button)
            if text:
                button.replace_with(LexborHTMLParser(f"<h4>{html_lib.escape(text)}</h4>").body.child)  # type: ignore[union-attr,arg-type]
        for node in tree.css(_REMOVE):
            node.decompose()

        tables: list[str] = []
        for table in tree.css("table"):
            if table.css_first("table") is not None:  # layout table: leave it to trafilatura
                continue
            markdown = _table_to_markdown(table)
            if not markdown:
                continue
            tables.append(markdown)
            placeholder = LexborHTMLParser(f"<p>RAGTABLE{len(tables) - 1}PLACEHOLDER</p>").body.child  # type: ignore[union-attr]
            table.replace_with(placeholder)  # type: ignore[arg-type]

        cleaned = tree.html or ""
        xml = trafilatura.extract(
            cleaned,
            output_format="xml",
            include_tables=True,
            include_comments=False,
            include_links=False,
            include_images=False,
            favor_recall=True,
            deduplicate=False,
        )
        blocks: list[ExtractedBlock] = []
        used: set[int] = set()
        tracker = SectionTracker()
        if xml:
            try:
                root = SafeET.fromstring(xml)
            except SafeET.ParseError as exc:
                raise AppError(ErrorCode.CORRUPT_FILE, "The HTML could not be parsed.") from exc
            main = root.find("main")
            for el in list(main if main is not None else root):
                tag = el.tag
                text = _xml_text(el)
                if not text:
                    continue
                if tag == "head":
                    level = _HEADING_TAGS.get(str(el.get("rend", "h2")), 2)
                    heading_path = tracker.enter(level, text)
                    blocks.append(ExtractedBlock(text=text, kind="heading", section_path=heading_path))
                elif tag == "p" and (m := _PLACEHOLDER.match(text)):
                    idx = int(m.group(1))
                    if idx < len(tables) and idx not in used:
                        used.add(idx)
                        blocks.append(
                            ExtractedBlock(text=tables[idx], kind="table", section_path=tracker.path)
                        )
                elif tag == "list":
                    items = [_xml_text(item) for item in el.iter("item")]
                    body = "\n".join(f"- {i}" for i in items if i)
                    if body:
                        blocks.append(ExtractedBlock(text=body, kind="list", section_path=tracker.path))
                elif tag == "table":  # a table trafilatura found itself (e.g. inside a layout table)
                    rows = [[_xml_text(c) for c in row.iter("cell")] for row in el.iter("row")]
                    markdown = rows_to_markdown(rows) if len(rows) >= 2 else ""
                    if markdown:
                        blocks.append(ExtractedBlock(text=markdown, kind="table", section_path=tracker.path))
                else:
                    text = _PLACEHOLDER.sub("", text).strip()
                    if text:
                        blocks.append(ExtractedBlock(text=text, kind="paragraph", section_path=tracker.path))
        # Tables trafilatura dropped as boilerplate are still content on government pages.
        for idx, markdown in enumerate(tables):
            if idx not in used:
                blocks.append(ExtractedBlock(text=markdown, kind="table", section_path=tracker.path))

        first_heading = next((b.text for b in blocks if b.kind == "heading"), None)
        if first_heading and (not title or len(title.split()) <= 2):  # "<title>NADRA</title>" is a site name
            title = first_heading[:300]
        metadata: dict[str, Any] = {}
        if source_url:
            metadata["source_url"] = source_url
        return ExtractedDocument(
            blocks=tuple(dedupe_blocks(blocks)),
            page_count=1,
            title=title,
            hidden_text_removed=hidden,
            metadata=metadata,
        )
