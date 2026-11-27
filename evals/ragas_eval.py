"""Optional RAGAS adapter (blueprint §10.4): faithfulness, factual correctness, context precision.

RAGAS is **optional**: it pulls a heavy dependency tree (datasets, pyarrow, pandas, …), so it lives in
`requirements-eval.in`, not in the runtime or dev requirements. When it is not installed this module
reports itself unavailable and `python -m evals.run --ragas` skips it with a message; the project's
own deterministic metrics and the rubric judge do not depend on it.

It is configured with the same judge (`settings.judge_model`) through LangChain's
`ChatOpenAI(base_url=LLM_BASE_URL)`, as the blueprint specifies. Its calls are **not** in the eval disk cache
and do not go through the harness's rate limiter, so run it on the baseline only, once the numbers
from the rest of the harness are final.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Mapping, Sequence
from typing import Any

from app.config import Settings
from evals.golden import GoldenItem


def is_available() -> bool:
    return (
        importlib.util.find_spec("ragas") is not None
        and importlib.util.find_spec("langchain_openai") is not None
    )


def _evaluate(
    records: Sequence[Mapping[str, Any]], items: Mapping[str, GoldenItem], settings: Settings
) -> dict[str, Any]:
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate  # type: ignore[import-not-found,unused-ignore]
    from ragas.dataset_schema import SingleTurnSample  # type: ignore[import-not-found,unused-ignore]
    from ragas.llms import LangchainLLMWrapper  # type: ignore[import-not-found,unused-ignore]
    from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference  # type: ignore[import-not-found,unused-ignore]

    samples = []
    for rec in records:
        item = items.get(str(rec.get("id")))
        if item is None or not item.answerable or rec.get("refused") or not rec.get("contexts"):
            continue
        samples.append(
            SingleTurnSample(
                user_input=item.question,
                response=str(rec.get("answer") or ""),
                retrieved_contexts=[str(c) for c in rec.get("contexts") or []],
                reference=item.reference_answer,
            )
        )
    if not samples:
        return {"n": 0}
    key = settings.llm_key or None
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=settings.judge_model, base_url=settings.llm_base_url, api_key=key, temperature=0)
    )
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), FactualCorrectness(), LLMContextPrecisionWithReference()],
        llm=llm,
        show_progress=False,
    )
    frame = result.to_pandas()
    out: dict[str, Any] = {"n": len(samples)}
    for column in frame.columns:
        if column in ("user_input", "response", "retrieved_contexts", "reference"):
            continue
        try:
            out[str(column)] = float(frame[column].mean())
        except (TypeError, ValueError):
            continue
    return out


async def score(
    records: Sequence[Mapping[str, Any]], items: Mapping[str, GoldenItem], settings: Settings
) -> dict[str, Any] | None:
    """RAGAS means over answered, answerable items, or None when RAGAS is not installed or fails."""
    if not is_available():
        return None
    try:
        return await asyncio.to_thread(_evaluate, records, items, settings)
    except Exception as exc:  # RAGAS is optional: never let it fail the run
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
