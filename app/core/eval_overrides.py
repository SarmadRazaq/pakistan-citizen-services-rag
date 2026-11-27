"""Per-request evaluation overrides: the `X-Eval-Config` header (EVAL.md "Configuration comparison").

The eval harness compares configurations by sending compact JSON on `/ask` and `/search`, e.g.
`{"config":"R1_dense_only","retrieval_mode":"dense","rerank":false}`. Supported keys:

- `config`: a label, echoed only;
- `retrieval_mode`: `dense` | `sparse` | `hybrid` (default `hybrid`);
- `rerank`: bool (default `true`);
- `generation_model`: one of `settings.generation_models`. The generate call then uses **only** that
  model (no fall-back across the rest of the chain, so a comparison never silently measures another
  model); the local fallback still applies when `LOCAL_FALLBACK_ENABLED`, and marks the answer degraded.

The header is honoured only when `EVAL_OVERRIDES_ENABLED=true` and the caller is an admin (see
`app.api.deps.get_eval_overrides`); this module only parses and validates. Anything invalid raises
`AppError(VALIDATION_FAILED)` (422).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from app.core.errors import AppError, ErrorCode

RetrievalMode = Literal["dense", "sparse", "hybrid"]
EVAL_HEADER = "X-Eval-Config"
APPLIED_HEADER = "X-Eval-Config-Applied"
MAX_HEADER_CHARS = 2048
MAX_LABEL_CHARS = 100
_KEYS = frozenset({"config", "retrieval_mode", "rerank", "generation_model"})


@dataclass(frozen=True, slots=True)
class EvalOverrides:
    """Validated overrides. Frozen and hashable, so it can sit inside a cache key."""

    config: str | None = None
    retrieval_mode: RetrievalMode = "hybrid"
    rerank: bool = True
    generation_model: str | None = None

    @property
    def is_default_retrieval(self) -> bool:
        return self.retrieval_mode == "hybrid" and self.rerank

    def applied_json(self) -> str:
        """The normalised overrides as compact JSON with sorted keys (the echo header)."""
        data: dict[str, Any] = {"retrieval_mode": self.retrieval_mode, "rerank": self.rerank}
        if self.config is not None:
            data["config"] = self.config
        if self.generation_model is not None:
            data["generation_model"] = self.generation_model
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _bad(detail: str) -> AppError:
    return AppError(ErrorCode.VALIDATION_FAILED, f"Invalid {EVAL_HEADER} header: {detail}")


def parse_eval_header(raw: str, allowed_models: Sequence[str]) -> EvalOverrides:
    if len(raw) > MAX_HEADER_CHARS:
        raise _bad("too long.")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise _bad("not valid JSON.") from exc
    if not isinstance(data, dict):
        raise _bad("expected a JSON object.")
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise _bad(f"unknown key(s) {', '.join(unknown)}.")

    config = data.get("config")
    if config is not None and not (isinstance(config, str) and 0 < len(config) <= MAX_LABEL_CHARS):
        raise _bad("config must be a non-empty string.")
    mode = data.get("retrieval_mode", "hybrid")
    if mode not in get_args(RetrievalMode):
        raise _bad("retrieval_mode must be dense, sparse or hybrid.")
    rerank = data.get("rerank", True)
    if not isinstance(rerank, bool):
        raise _bad("rerank must be true or false.")
    model = data.get("generation_model")
    if model is not None and (not isinstance(model, str) or model not in allowed_models):
        raise _bad("generation_model is not one of the configured generation models.")
    return EvalOverrides(config=config, retrieval_mode=mode, rerank=rerank, generation_model=model)
