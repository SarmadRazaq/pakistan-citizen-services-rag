"""Render `report.md` from `metrics.json`, and refresh the generated tables in `EVAL.md`.

EVAL.md keeps the student's prose; only the blocks between markers are regenerated:

    <!-- EVAL:START config_comparison -->
    ...generated...
    <!-- EVAL:END config_comparison -->

A bare `<!-- EVAL:START -->` / `<!-- EVAL:END -->` pair receives the whole report.

Re-render an existing run (no network):  python -m evals.report evals/results/<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.metrics import ItemScore

REPO = Path(__file__).resolve().parents[1]
EVAL_MD = REPO / "EVAL.md"

BLOCKS = (
    "run_info",
    "golden_set",
    "config_comparison",
    "by_type",
    "multilingual",
    "contamination",
    "failure_analysis",
)

Column = tuple[str, str, str]  # (metric key, header, format)

SERVICE_COLUMNS: list[Column] = [
    ("n", "n", "int"),
    ("hit@1", "Hit@1", "f"),
    ("hit@5", "Hit@5", "f"),
    ("hit@10", "Hit@10", "f"),
    ("recall@10", "R@10", "f"),
    ("mrr@10", "MRR@10", "f"),
    ("ndcg@10", "nDCG@10", "f"),
    ("refusal_f1", "Refusal F1", "f"),
    ("false_refusal_rate", "False refusal", "f"),
    ("citation_validity", "Citation valid", "f"),
    ("notices_ok", "Notices", "f"),
    ("language_match", "Lang match", "f"),
    ("judge_correctness_mean", "Judge corr. (0-2)", "f2"),
    ("judge_correct_rate", "Judge = 2", "f"),
    ("latency_p50_ms", "p50 ms", "int"),
    ("latency_p95_ms", "p95 ms", "int"),
]

GROUP_COLUMNS: list[Column] = [
    ("n", "n", "int"),
    ("hit@5", "Hit@5", "f"),
    ("mrr@10", "MRR@10", "f"),
    ("refusal_f1", "Refusal F1", "f"),
    ("citation_validity", "Citation valid", "f"),
    ("language_match", "Lang match", "f"),
    ("judge_correctness_mean", "Judge corr. (0-2)", "f2"),
]

CSV_FIELDS = [
    "config",
    "id",
    "type",
    "language",
    "answer_language",
    "split",
    "answerable",
    "hit@1",
    "hit@3",
    "hit@5",
    "hit@10",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@10",
    "context_hit",
    "refused",
    "refusal_reason",
    "citation_valid",
    "n_citations",
    "expected_notices",
    "notices",
    "notices_ok",
    "expected_language",
    "detected_language",
    "language_ok",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "model",
    "judge_correctness",
    "judge_citation_support",
    "judge_refusal_appropriate",
    "judge_advice_ok",
    "failure_category",
    "error",
    "judge_rationale",
]


# ---------------------------------------------------------------- formatting helpers


def fmt(value: Any, kind: str = "f") -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if kind == "int":
        return str(round(float(value)))
    if kind == "f2":
        return f"{float(value):.2f}"
    if kind == "f":
        return f"{float(value):.3f}"
    return str(value)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _configs(metrics: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    return list((metrics.get("configs") or {}).items())


def _override_status(meta: Mapping[str, Any]) -> str:
    return {
        "n/a": "none",
        "yes": "honoured",
        "no": "NOT honoured (service defaults)",
        "unknown": "unknown",
    }.get(str(meta.get("override_honoured")), str(meta.get("override_honoured")))


# ---------------------------------------------------------------- tables


def comparison_table(metrics: Mapping[str, Any]) -> str:
    headers = ["Config", *[c[1] for c in SERVICE_COLUMNS], "Overrides"]
    rows = []
    for name, data in _configs(metrics):
        meta = data.get("meta", {})
        overall = data.get("overall", {})
        label = f"**{name}** (baseline)" if meta.get("baseline") else name
        cells = [fmt(overall.get(key), kind) for key, _, kind in SERVICE_COLUMNS]
        if meta.get("kind") == "closed_book":
            # no retrieval, citations or refusals in a closed-book answer
            keep = {"n", "language_match", "judge_correctness_mean", "judge_correct_rate"}
            cells = [c if key in keep else "–" for c, (key, _, _) in zip(cells, SERVICE_COLUMNS, strict=True)]
            override = "closed-book (no sources)"
        else:
            override = _override_status(meta)
        rows.append([label, *cells, override])
    return md_table(headers, rows)


def _group_table(metrics: Mapping[str, Any], group: str, order: Sequence[str] | None = None) -> str:
    headers = ["Config", group.removeprefix("by_").capitalize(), *[c[1] for c in GROUP_COLUMNS]]
    rows = []
    for name, data in _configs(metrics):
        groups: Mapping[str, Any] = data.get(group) or {}
        keys = [k for k in (order or sorted(groups)) if k in groups]
        for key in keys:
            rows.append([name, key, *[fmt(groups[key].get(m), kind) for m, _, kind in GROUP_COLUMNS]])
    return md_table(headers, rows) if rows else "_No items._"


def language_table(metrics: Mapping[str, Any]) -> str:
    table = _group_table(metrics, "by_language", ["en", "ur", "roman_ur"])
    gap_rows = []
    for name, data in _configs(metrics):
        groups = data.get("by_language") or {}
        en = groups.get("en") or {}
        for lang in ("ur", "roman_ur"):
            other = groups.get(lang)
            if not other:
                continue
            cells = []
            for key in ("hit@5", "judge_correctness_mean", "language_match"):
                a, b = other.get(key), en.get(key)
                cells.append(f"{a - b:+.3f}" if a is not None and b is not None else "–")
            gap_rows.append([name, lang, *cells])
    if not gap_rows:
        return table
    gap = md_table(
        ["Config", "Language", "Δ Hit@5 vs en", "Δ Judge corr. vs en", "Δ Lang match vs en"], gap_rows
    )
    return (
        f"{table}\n\n**Cross-lingual gap** (language group minus the English group; the blueprint's "
        f"paired version needs translation links in the golden set):\n\n{gap}"
    )


def contamination_table(metrics: Mapping[str, Any]) -> str:
    rows = []
    closed = [(n, d) for n, d in _configs(metrics) if d.get("meta", {}).get("kind") == "closed_book"]
    rag = [(n, d) for n, d in _configs(metrics) if d.get("meta", {}).get("kind") != "closed_book"]
    if not closed:
        return "_No closed-book config in this run._"
    baseline = next((d for _, d in rag if d.get("meta", {}).get("baseline")), rag[0][1] if rag else None)
    base_corr = (baseline or {}).get("overall", {}).get("judge_correctness_mean")
    for name, data in closed + rag:
        overall = data.get("overall", {})
        corr = overall.get("judge_correctness_mean")
        is_closed = data.get("meta", {}).get("kind") == "closed_book"
        gain = f"{base_corr - corr:+.2f}" if is_closed and corr is not None and base_corr is not None else ""
        rows.append(
            [
                name,
                str(data.get("meta", {}).get("model") or "–"),
                fmt(overall.get("n_answerable"), "int"),
                fmt(corr, "f2"),
                fmt(overall.get("judge_correct_rate")),
                gain,
            ]
        )
    return md_table(
        ["Config", "Model", "Answerable n", "Judge corr. (0-2)", "Judge = 2", "RAG baseline − closed-book"],
        rows,
    )


def failure_table(metrics: Mapping[str, Any], *, limit: int = 60) -> str:
    rows = []
    counts = []
    for name, data in _configs(metrics):
        failures = data.get("failures") or {}
        counts.append(
            [
                name,
                ", ".join(f"{k}: {v}" for k, v in failures.items() if k != "ok") or "none",
                str(failures.get("ok", 0)),
            ]
        )
        for item in data.get("failed_items") or []:
            rows.append(
                [
                    name,
                    item.get("id", ""),
                    item.get("type", ""),
                    item.get("language", ""),
                    item.get("category", ""),
                    (item.get("detail") or "")[:160],
                    "",
                ]
            )
    summary = md_table(["Config", "Failures by category", "ok"], counts)
    if not rows:
        return summary + "\n\n_No failed items._"
    detail = md_table(
        [
            "Config",
            "Item",
            "Type",
            "Lang",
            "Category",
            "Detail (judge rationale / error)",
            "Root cause & fix",
        ],
        rows[:limit],
    )
    more = f"\n\n_{len(rows) - limit} more in per_question.csv._" if len(rows) > limit else ""
    return f"{summary}\n\n{detail}{more}"


def golden_table(metrics: Mapping[str, Any]) -> str:
    run = metrics.get("run") or {}
    golden = run.get("golden") or {}
    sha = str(golden.get("sha256", ""))[:16]
    drafts = "included" if run.get("include_drafts") else "excluded"
    parts = [
        f"File `{golden.get('path', 'evals/golden_set.jsonl')}`, sha256 `{sha}…`: "
        f"**{golden.get('total', 0)} items** in the file, **{golden.get('evaluated', 0)} evaluated** "
        f"in this run (split `{run.get('split')}`, drafts {drafts})."
    ]
    for key, title in (
        ("by_type", "Type"),
        ("by_language", "Language"),
        ("by_split", "Split"),
        ("by_status", "Review status"),
    ):
        counts: Mapping[str, int] = golden.get(key) or {}
        if counts:
            parts.append(md_table([title, "Items"], [[k, str(v)] for k, v in counts.items()]))
    return "\n\n".join(parts)


def run_info(metrics: Mapping[str, Any]) -> str:
    run = metrics.get("run") or {}
    lines = [
        f"- Run: `{run.get('name')}` at {run.get('started_at')} against `{run.get('base_url')}`"
        f"{' (offline, from caches)' if run.get('offline') else ''}",
        f"- Judge: `{run.get('judge_model')}` (prompt `{run.get('judge_prompt_version')}`), "
        f"{'enabled' if run.get('judge') else 'disabled'}",
        f"- Configs: {', '.join(f'`{c}`' for c in run.get('configs', []))}",
        f"- Command: `{run.get('command')}`",
    ]
    return "\n".join(lines)


def render_report(metrics: Mapping[str, Any]) -> str:
    sections = [
        "# Evaluation report",
        run_info(metrics),
        "## Golden set",
        golden_table(metrics),
        "## Configuration comparison",
        comparison_table(metrics),
        "_Overrides_: a config whose overrides were **not honoured** ran on the service defaults "
        "(the service does not read `X-Eval-Config` yet), so its row repeats the baseline.",
        "## By question type",
        _group_table(metrics, "by_type"),
        "## Multilingual (reported separately)",
        language_table(metrics),
        "## By split",
        _group_table(metrics, "by_split", ["dev", "test"]),
        "## Contamination (closed-book baseline)",
        contamination_table(metrics),
        "## Failure analysis",
        failure_table(metrics),
    ]
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------- EVAL.md markers


def _markers(name: str) -> tuple[str, str]:
    suffix = f" {name}" if name else ""
    return f"<!-- EVAL:START{suffix} -->", f"<!-- EVAL:END{suffix} -->"


def replace_block(text: str, name: str, content: str) -> tuple[str, bool]:
    """Replace what sits between the START/END markers of `name`. Returns (text, replaced?)."""
    start, end = _markers(name)
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda _: f"{start}\n{content.strip()}\n{end}", text, count=1), True


BLOCK_RENDERERS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "run_info": run_info,
    "golden_set": golden_table,
    "config_comparison": comparison_table,
    "by_type": lambda m: _group_table(m, "by_type"),
    "multilingual": language_table,
    "contamination": contamination_table,
    "failure_analysis": failure_table,
}


def update_eval_md(path: Path, metrics: Mapping[str, Any]) -> list[str]:
    """Refresh every generated block present in `path`. Returns the names of the blocks replaced."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    replaced = []
    for name, render in BLOCK_RENDERERS.items():
        text, ok = replace_block(text, name, render(metrics))
        if ok:
            replaced.append(name)
    text, ok = replace_block(text, "", render_report(metrics))
    if ok:
        replaced.append("(whole report)")
    path.write_text(text, encoding="utf-8")
    return replaced


# ---------------------------------------------------------------- per_question.csv


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    return str(value)


def score_row(score: ItemScore) -> dict[str, str]:
    data = score.to_dict()
    data.update(score.retrieval or {})
    return {key: _cell(data.get(key)) for key in CSV_FIELDS}


def write_per_question_csv(path: Path, scores: Sequence[ItemScore]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for score in scores:
            writer.writerow(score_row(score))


def read_per_question_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-render report.md and EVAL.md from a results folder")
    parser.add_argument("results", type=Path, help="evals/results/<timestamp>")
    parser.add_argument("--eval-md", type=Path, default=EVAL_MD)
    parser.add_argument("--no-eval-md", action="store_true")
    args = parser.parse_args(argv)
    metrics = json.loads((args.results / "metrics.json").read_text(encoding="utf-8"))
    (args.results / "report.md").write_text(render_report(metrics), encoding="utf-8")
    print(f"wrote {args.results / 'report.md'}")
    if not args.no_eval_md:
        blocks = update_eval_md(args.eval_md, metrics)
        print(f"updated {args.eval_md}: {', '.join(blocks) or 'no markers found'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
