"""Judge calibration against human labels (blueprint §10.5, step 1).

1. Export a stratified sample of 30 answers for the student to label, **without** the judge's scores:
       python -m evals.calibration export --results evals/results/<timestamp>
   This writes evals/human_labels.csv. Fill in, for every row:
       human_correctness        2 = correct, 1 = partial, 0 = wrong   (or: correct / partial / wrong)
       human_citation_supported y / n   (leave empty when there are no citations or it refused)
2. Score the judge against those labels:
       python -m evals.calibration score --results evals/results/<timestamp>
   Prints % agreement and Cohen's κ (target κ ≥ 0.6), overall and on the multilingual rows, writes
   calibration.json into the results folder and refreshes the `judge_calibration` block in EVAL.md.

Cohen's κ is implemented here in plain Python: κ = (p_o − p_e) / (1 − p_e).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evals import report

EVALS = Path(__file__).resolve().parent
HUMAN_LABELS = EVALS / "human_labels.csv"
LABEL_FIELDS = [
    "key",
    "config",
    "id",
    "language",
    "type",
    "answerable",
    "question",
    "reference_answer",
    "answer",
    "refused",
    "citations",
    "human_correctness",
    "human_citation_supported",
    "labeller_notes",
]


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's κ for two raters over the same items. None when there are no items."""
    if len(a) != len(b):
        raise ValueError("both raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    labels = set(a) | set(b)
    p_o = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    p_e = sum((list(a).count(lab) / n) * (list(b).count(lab) / n) for lab in labels)
    if p_e == 1.0:  # both raters used one single label throughout
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def percent_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    if not a:
        return None
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


# ---------------------------------------------------------------- sampling


def stratified_sample(
    rows: Sequence[Mapping[str, Any]], n: int = 30, seed: int = 13
) -> list[Mapping[str, Any]]:
    """Round-robin over (language, type) strata, shuffled with a fixed seed, so every language and
    question type is represented and the multilingual rows (blueprint: ≥ 8) get their share."""
    rng = random.Random(seed)  # noqa: S311 (reproducible sampling, not crypto)
    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        strata.setdefault((str(row.get("language")), str(row.get("type"))), []).append(row)
    for group in strata.values():
        rng.shuffle(group)
    order = sorted(strata)
    picked: list[Mapping[str, Any]] = []
    while len(picked) < n and any(strata[k] for k in order):
        for key in order:
            if strata[key] and len(picked) < n:
                picked.append(strata[key].pop())
    return picked


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _baseline(metrics: Mapping[str, Any]) -> str | None:
    service = [(n, d) for n, d in (metrics.get("configs") or {}).items() if d["meta"]["kind"] == "service"]
    for name, data in service:
        if data["meta"].get("baseline"):
            return str(name)
    return service[0][0] if service else None


def export_labels(
    results: Path, out: Path = HUMAN_LABELS, *, n: int = 30, config: str | None = None, force: bool = False
) -> int:
    if out.exists() and not force and _has_labels(out):
        raise FileExistsError(f"{out} already has human labels; pass --force to overwrite them")
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    config = config or _baseline(metrics)
    meta = {
        r["id"]: r
        for r in report.read_per_question_csv(results / "per_question.csv")
        if r["config"] == config
    }
    answers = [
        a for a in _read_jsonl(results / "answers.jsonl") if a.get("config") == config and "answer" in a
    ]
    golden = {i["id"]: i for i in _golden_items(metrics)}
    rows = []
    for a in answers:
        info = meta.get(a["id"], {})
        item = golden.get(a["id"], {})
        rows.append(
            {
                "key": f"{config}|{a['id']}",
                "config": config,
                "id": a["id"],
                "language": info.get("language", item.get("language", "")),
                "type": info.get("type", item.get("type", "")),
                "answerable": info.get("answerable", ""),
                "question": a.get("question", ""),
                "reference_answer": item.get("reference_answer", ""),
                "answer": a.get("answer", ""),
                "refused": "1" if a.get("refused") else "0",
                "citations": " | ".join(
                    f"[{c.get('marker')}] {c.get('file_name')} p.{c.get('page')}: {c.get('quote', '')}"
                    for c in a.get("citations") or []
                ),
                "human_correctness": "",
                "human_citation_supported": "",
                "labeller_notes": "",
            }
        )
    sample = stratified_sample(rows, n=n)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel shows Urdu correctly
        writer = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(sample)
    return len(sample)


def _golden_items(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The golden set the run used (its path is recorded in metrics.json), for the reference answers."""
    from evals.golden import GOLDEN_PATH, read_items

    recorded = ((metrics.get("run") or {}).get("golden") or {}).get("path")
    path = Path(recorded) if recorded else GOLDEN_PATH
    if not path.is_absolute():
        path = EVALS.parent / path
    return [i.model_dump() for i in read_items(path if path.exists() else GOLDEN_PATH)]


def _has_labels(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return any((row.get("human_correctness") or "").strip() for row in csv.DictReader(fh))


# ---------------------------------------------------------------- scoring

_CORRECTNESS = {"2": 2, "1": 1, "0": 0, "correct": 2, "partial": 1, "wrong": 0, "incorrect": 0}
_SUPPORT = {
    "y": "yes",
    "yes": "yes",
    "1": "yes",
    "true": "yes",
    "n": "no",
    "no": "no",
    "0": "no",
    "false": "no",
}


def score_labels(results: Path, labels: Path = HUMAN_LABELS) -> dict[str, Any]:
    judge = {
        f"{r['config']}|{r['id']}": r for r in report.read_per_question_csv(results / "per_question.csv")
    }
    with labels.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    pairs: list[tuple[int, int, bool]] = []
    support: list[tuple[str, str, bool]] = []
    missing = 0
    for row in rows:
        human = _CORRECTNESS.get((row.get("human_correctness") or "").strip().casefold())
        j = judge.get(row.get("key", ""))
        if human is None:
            continue
        if not j or not j.get("judge_correctness"):
            missing += 1
            continue
        multilingual = row.get("language") != "en"
        pairs.append((human, int(j["judge_correctness"]), multilingual))
        hs = _SUPPORT.get((row.get("human_citation_supported") or "").strip().casefold())
        js = j.get("judge_citation_support")
        if hs and js in ("yes", "no"):
            support.append((hs, js, multilingual))

    def stats(data: Sequence[tuple[Any, Any, bool]]) -> dict[str, Any]:
        a, b = [x[0] for x in data], [x[1] for x in data]
        return {"n": len(data), "agreement": percent_agreement(a, b), "kappa": cohens_kappa(a, b)}

    return {
        "labels_file": labels.as_posix(),
        "results": results.as_posix(),
        "unjudged_labelled_rows": missing,
        "correctness": stats(pairs),
        "correctness_multilingual": stats([p for p in pairs if p[2]]),
        "correctness_binary": stats([(h == 2, j == 2, m) for h, j, m in pairs]),
        "citation_support": stats(support),
        "target_kappa": 0.6,
    }


def calibration_table(cal: Mapping[str, Any]) -> str:
    rows = []
    for key, title in (
        ("correctness", "Correctness 0/1/2"),
        ("correctness_binary", "Correct (2) vs not"),
        ("correctness_multilingual", "Correctness, Urdu + Roman Urdu only"),
        ("citation_support", "Citation supported y/n"),
    ):
        s = cal.get(key) or {}
        rows.append([title, str(s.get("n", 0)), report.fmt(s.get("agreement")), report.fmt(s.get("kappa"))])
    return report.md_table(["Judge vs human", "n", "% agreement", "Cohen's κ"], rows) + (
        f"\n\nTarget κ ≥ {cal.get('target_kappa', 0.6)}; below it, the rubric in `evals/judge.py` is revised "
        f"and the run repeated. Labels: `{cal.get('labels_file')}`; run: `{cal.get('results')}`."
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.calibration", description="Judge vs human labels")
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write a stratified sample to label")
    ex.add_argument("--results", type=Path, required=True)
    ex.add_argument("--out", type=Path, default=HUMAN_LABELS)
    ex.add_argument("--n", type=int, default=30)
    ex.add_argument("--config", default=None, help="default: the baseline config")
    ex.add_argument("--force", action="store_true", help="overwrite a file that already has labels")
    sc = sub.add_parser("score", help="agreement and Cohen's kappa")
    sc.add_argument("--results", type=Path, required=True)
    sc.add_argument("--labels", type=Path, default=HUMAN_LABELS)
    sc.add_argument("--eval-md", type=Path, default=report.EVAL_MD)
    sc.add_argument("--no-eval-md", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "export":
        count = export_labels(args.results, args.out, n=args.n, config=args.config, force=args.force)
        print(
            f"wrote {count} rows to {args.out}. Label human_correctness (2/1/0) and "
            "human_citation_supported (y/n) without looking at per_question.csv."
        )
        return 0
    cal = score_labels(args.results, args.labels)
    (args.results / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    table = calibration_table(cal)
    print(table)
    if not args.no_eval_md and args.eval_md.exists():
        text, ok = report.replace_block(args.eval_md.read_text(encoding="utf-8"), "judge_calibration", table)
        if ok:
            args.eval_md.write_text(text, encoding="utf-8")
            print(f"updated {args.eval_md} (judge_calibration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
