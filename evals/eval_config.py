"""Eval configurations (`evals/configs/*.yaml`, blueprint §10.6).

A config sets what the harness can vary **through the public API**:
- request params: `top_k` (/ask), `search_top_k` (/search), `answer_language`, `scope`;
- `overrides`: retrieval/generation overrides sent as a JSON `X-Eval-Config` header (keys `config`,
  `retrieval_mode`, `rerank`, `generation_model`; see `app/core/eval_overrides.py`). The service honours
  the header when `EVAL_OVERRIDES_ENABLED=true` **and** the eval account is an admin, and then echoes the
  applied overrides in `X-Eval-Config-Applied`. Otherwise the run falls back to the service's default
  pipeline, and the report flags the config as not honoured.

`kind: closed_book` is the contamination baseline: the harness asks the generator model directly via
the NVIDIA API with **no sources**, and the judge scores it like any other answer.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped,unused-ignore]
from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
EVAL_HEADER = "X-Eval-Config"
APPLIED_HEADER = "X-Eval-Config-Applied"


class RequestParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=5, ge=1, le=10)
    search_top_k: int = Field(default=20, ge=10, le=50)  # ≥ 10 so Hit@10 / nDCG@10 are defined
    answer_language: Literal["auto", "item", "en", "ur", "roman_ur"] = "auto"
    # Golden items are judged against the Official corpus only (GLOSSARY.md "Golden item"), so a config
    # can never widen the scope to the eval user's own uploads.
    scope: Literal["public"] = "public"


class ClosedBookParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None  # default: the first of settings.generation_models
    max_tokens: int = 700


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    description: str = ""
    kind: Literal["service", "closed_book"] = "service"
    default: bool = True  # part of a run when --configs is not given
    baseline: bool = False  # the reference row: contamination gap and calibration sample use it
    request: RequestParams = Field(default_factory=RequestParams)
    overrides: dict[str, Any] = Field(default_factory=dict)
    closed_book: ClosedBookParams = Field(default_factory=ClosedBookParams)
    source: str | None = None

    @property
    def needs_service_support(self) -> bool:
        return self.kind == "service" and bool(self.overrides)

    def header_value(self) -> str | None:
        """Compact, ASCII-only JSON for the X-Eval-Config header (None = send no header)."""
        if not self.overrides:
            return None
        return json.dumps({"config": self.name, **self.overrides}, sort_keys=True, separators=(",", ":"))


def load_config(path: Path) -> EvalConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("name", path.stem)
    data["source"] = path.as_posix()
    return EvalConfig.model_validate(data)


def resolve_configs(specs: list[str] | None, config_dir: Path = CONFIG_DIR) -> list[EvalConfig]:
    """Paths, globs (expanded here, since PowerShell does not) or bare names like `R4_hybrid_rerank`.

    With no specs, every config in `config_dir` marked `default: true`.
    """
    if not specs:
        configs = [load_config(p) for p in sorted(config_dir.glob("*.yaml"))]
        # baseline first, then the other service configs, the closed-book baseline last
        configs.sort(key=lambda c: (not c.baseline, c.kind == "closed_book", c.name))
        return [c for c in configs if c.default]
    paths: list[Path] = []
    for spec in specs:
        if any(ch in spec for ch in "*?["):
            matched = sorted(Path(p) for p in glob.glob(spec))  # noqa: PTH207 (absolute globs too)
            if not matched:
                raise FileNotFoundError(f"no config matches {spec!r}")
            paths.extend(matched)
            continue
        candidate = Path(spec)
        if not candidate.exists():
            candidate = config_dir / (spec if spec.endswith(".yaml") else f"{spec}.yaml")
        if not candidate.exists():
            raise FileNotFoundError(f"config {spec!r} not found (looked in {config_dir})")
        paths.append(candidate)
    seen: set[str] = set()
    configs = []
    for p in paths:
        cfg = load_config(p)
        if cfg.name in seen:
            continue
        seen.add(cfg.name)
        configs.append(cfg)
    return configs
