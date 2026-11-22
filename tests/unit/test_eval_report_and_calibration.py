"""EVAL.md marker replacement, golden loading, config loading and Cohen's kappa (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evals.calibration import cohens_kappa, percent_agreement, stratified_sample
from evals.eval_config import CONFIG_DIR, resolve_configs
from evals.golden import GOLDEN_PATH, load_golden, read_items
from evals.report import replace_block, update_eval_md

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- EVAL.md markers


def test_replace_named_block_keeps_surrounding_prose() -> None:
    text = (
        "# Eval\nprose before\n<!-- EVAL:START config_comparison -->\nold table\n"
        "<!-- EVAL:END config_comparison -->\nprose after\n"
    )
    new, ok = replace_block(text, "config_comparison", "| a |\n|---|")
    assert ok
    assert new == (
        "# Eval\nprose before\n<!-- EVAL:START config_comparison -->\n| a |\n|---|\n"
        "<!-- EVAL:END config_comparison -->\nprose after\n"
    )
    again, _ = replace_block(new, "config_comparison", "| a |\n|---|")
    assert again == new  # idempotent


def test_replace_plain_block_and_missing_block() -> None:
    text = "x\n<!-- EVAL:START -->\n\n<!-- EVAL:END -->\ny"
    new, ok = replace_block(text, "", "report")
    assert ok and new == "x\n<!-- EVAL:START -->\nreport\n<!-- EVAL:END -->\ny"
    same, ok = replace_block(text, "multilingual", "z")
    assert not ok and same == text


def test_replacement_text_with_backslashes_is_literal() -> None:
    new, _ = replace_block("<!-- EVAL:START a -->\n<!-- EVAL:END a -->", "a", r"C:\path \1 \g<0>")
    assert r"C:\path \1 \g<0>" in new


def test_update_eval_md_fills_every_block(tmp_path: Path) -> None:
    eval_md = tmp_path / "EVAL.md"
    eval_md.write_text((REPO / "EVAL.md").read_text(encoding="utf-8"), encoding="utf-8")
    metrics: dict[str, Any] = {
        "run": {
            "name": "t",
            "configs": ["R4"],
            "golden": {"total": 1, "evaluated": 1, "by_type": {"single_fact": 1}},
        },
        "configs": {
            "R4": {
                "meta": {"kind": "service", "baseline": True, "override_honoured": "n/a"},
                "overall": {"n": 1, "hit@5": 1.0, "mrr@10": 0.5},
                "by_type": {"single_fact": {"n": 1, "hit@5": 1.0}},
                "by_language": {"en": {"n": 1, "hit@5": 1.0}},
                "by_split": {"test": {"n": 1}},
                "failures": {"ok": 1},
                "failed_items": [],
            }
        },
    }
    replaced = update_eval_md(eval_md, metrics)
    assert {
        "run_info",
        "golden_set",
        "config_comparison",
        "by_type",
        "multilingual",
        "contamination",
        "failure_analysis",
    } <= set(replaced)
    text = eval_md.read_text(encoding="utf-8")
    assert "**R4** (baseline)" in text and "| 0.500 |" in text
    assert "<!-- EVAL:START judge_calibration -->" in text  # calibration block left for calibration.py
    assert "## Generator comparison" in text  # prose untouched


# ---------------------------------------------------------------- golden + configs


def test_load_golden_filters_drafts_and_split() -> None:
    everything = read_items(GOLDEN_PATH)
    approved = [i for i in everything if i.review_status == "approved"]
    assert len(load_golden(GOLDEN_PATH)) == len(approved)
    drafts = load_golden(GOLDEN_PATH, include_drafts=True)
    assert len(drafts) == len([i for i in everything if i.review_status != "dropped"])
    dev = load_golden(GOLDEN_PATH, include_drafts=True, split="dev")
    assert dev and all(i.split == "dev" for i in dev)


def test_golden_item_rules(tmp_path: Path) -> None:
    from evals.golden import GoldenItem

    line = json.loads(GOLDEN_PATH.read_text(encoding="utf-8").splitlines()[0])
    line.update(answerable=False)
    with pytest.raises(ValueError, match="gold sources"):
        GoldenItem.model_validate(line)


def test_shipped_configs() -> None:
    default = {c.name: c for c in resolve_configs(None, CONFIG_DIR)}
    assert list(default) == ["R4_hybrid_rerank", "R1_dense_only", "CB_closed_book"]  # baseline first
    assert default["R4_hybrid_rerank"].baseline and not default["R4_hybrid_rerank"].needs_service_support
    assert default["R1_dense_only"].needs_service_support
    header = default["R1_dense_only"].header_value()
    assert header is not None and json.loads(header)["retrieval_mode"] == "dense"
    assert default["CB_closed_book"].kind == "closed_book"
    named = resolve_configs(["R5_hybrid_no_rerank", str(CONFIG_DIR / "R4_*.yaml")], CONFIG_DIR)
    assert [c.name for c in named] == ["R5_hybrid_no_rerank", "R4_hybrid_rerank"]
    with pytest.raises(FileNotFoundError):
        resolve_configs(["nope"], CONFIG_DIR)


# ---------------------------------------------------------------- kappa


def test_cohens_kappa_known_example() -> None:
    # Classic 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> p_o 0.7, p_e 0.5, kappa 0.4
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert percent_agreement(a, b) == pytest.approx(0.7)
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_cohens_kappa_edges() -> None:
    assert cohens_kappa([2, 1, 0], [2, 1, 0]) == pytest.approx(1.0)
    assert cohens_kappa([], []) is None
    assert cohens_kappa([1, 1], [1, 1]) == 1.0  # one label throughout, full agreement
    assert cohens_kappa([0, 1], [1, 0]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 2])


def test_stratified_sample_covers_every_stratum() -> None:
    rows = [
        {"id": f"q{i}", "language": lang, "type": t}
        for i, (lang, t) in enumerate(
            [("en", "single_fact")] * 40 + [("ur", "single_fact")] * 5 + [("roman_ur", "advice_boundary")] * 5
        )
    ]
    sample = stratified_sample(rows, n=12)
    assert len(sample) == 12
    langs = [r["language"] for r in sample]
    assert langs.count("ur") == 4 and langs.count("roman_ur") == 4  # round-robin, not proportional
    assert stratified_sample(rows, n=12) == sample  # seeded, reproducible
