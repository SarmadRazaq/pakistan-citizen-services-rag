"""Check the LLM key and the configured model ids (feedback item 2).

Usage:  .venv\\Scripts\\python scripts\\check_models.py [--probe]

- Lists every model the LLM key can see (GET {LLM_BASE_URL}/models; NVIDIA, OpenAI or any
  OpenAI-compatible base).
- Reports which of GENERATION_MODELS / HELPER_MODEL / JUDGE_MODEL are missing.
- With --probe, sends one tiny JSON-mode request per configured model and reports
  latency, whether JSON mode was accepted, and whether the reply parsed as JSON.
- Checks exactly the retrieval providers configured: the NVIDIA embedder / reranker, or the OpenAI
  embedder (EMBEDDING_PROVIDER=openai). Local providers and RERANKER_PROVIDER=none are not called;
  the output lists what was skipped and why. Each request carries only its own provider's key.
- Writes the result to evals/configs/models_available.json (no key inside), so the
  evaluation records exactly which models existed on the day it ran.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings

OUT = Path("evals/configs/models_available.json")


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def _probe(client: httpx.Client, base: str, model: str, settings: Settings, key: str) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": 'Reply with JSON only: {"ok": true, "lang": "<language of the user>"}',
            },
            {"role": "user", "content": "Passport ki fee kitni hai?"},
        ],
        "response_format": {"type": "json_object"},
    }
    if settings.llm_token_param == "max_completion_tokens":  # noqa: S105
        body["max_completion_tokens"] = 300  # reasoning models think first; they also reject temperature
    else:
        body["max_tokens"] = 60
        body["temperature"] = 0
    headers = _bearer(key)
    started = time.perf_counter()
    r = client.post(f"{base}/chat/completions", json=body, headers=headers)
    json_mode = True
    if r.status_code in (400, 422):  # model rejects response_format: retry without it
        json_mode = False
        body.pop("response_format")
        r = client.post(f"{base}/chat/completions", json=body, headers=headers)
    latency = round((time.perf_counter() - started) * 1000)
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "latency_ms": latency, "json_mode": json_mode}
    text = r.json()["choices"][0]["message"].get("content") or ""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        json.loads(text[start:end])
        parsed = True
    except ValueError:
        parsed = False
    return {"ok": True, "latency_ms": latency, "json_mode": json_mode, "json_parsed": parsed}


Request = tuple[str, str, dict[str, object], dict[str, str], bool]  # model, url, body, headers, listed?


def _retrieval_requests(settings: Settings, skipped: dict[str, str]) -> dict[str, Request]:
    requests: dict[str, Request] = {}
    if settings.embedding_provider == "nvidia":
        if settings.nvidia_key:
            body: dict[str, object] = {
                "model": settings.hosted_embedding_model,
                "input": ["passport"],
                "input_type": "query",
                "encoding_format": "float",
                "truncate": "END",
            }
            url = f"{settings.nvidia_base_url.rstrip('/')}/embeddings"
            requests["embedding"] = (
                settings.hosted_embedding_model,
                url,
                body,
                _bearer(settings.nvidia_key),
                True,
            )
        else:
            skipped["embedding"] = "NVIDIA embedder selected but no NVIDIA key is set"
    elif settings.embedding_provider == "openai":
        if settings.openai_key:
            body = {
                "model": settings.openai_embedding_model,
                "input": ["passport"],
                "encoding_format": "float",
                "dimensions": settings.embedding_dimension,
            }
            url = f"{settings.openai_base_url.rstrip('/')}/embeddings"
            requests["embedding"] = (
                settings.openai_embedding_model,
                url,
                body,
                _bearer(settings.openai_key),
                False,
            )
        else:
            skipped["embedding"] = "OpenAI embedder selected but OPENAI_API_KEY is not set"
    else:
        skipped["embedding"] = "EMBEDDING_PROVIDER=local (runs on this machine, nothing to call)"

    if settings.reranker_provider == "nvidia":
        if settings.nvidia_key:
            rerank_body: dict[str, object] = {
                "model": settings.hosted_reranker_model,
                "query": {"text": "passport fee"},
                "passages": [{"text": "The passport fee is listed on the office page."}],
                "truncate": "END",
            }
            requests["reranker"] = (
                settings.hosted_reranker_model,
                settings.reranker_url,
                rerank_body,
                _bearer(settings.nvidia_key),
                True,
            )
        else:
            skipped["reranker"] = "NVIDIA reranker selected but no NVIDIA key is set"
    elif settings.reranker_provider == "none":
        skipped["reranker"] = "RERANKER_PROVIDER=none (retrieval gates on dense similarity)"
    else:
        skipped["reranker"] = "RERANKER_PROVIDER=local (runs on this machine, nothing to call)"

    skipped["injection"] = (
        "INJECTION_PROVIDER=none (heuristic guards only)"
        if settings.injection_provider == "none"
        else "INJECTION_PROVIDER=nvidia (classifier not probed here)"
    )
    return requests


def check_retrieval(
    client: httpx.Client, settings: Settings, available: list[str]
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Visibility and reachability of the hosted retrieval models actually configured.

    Returns `(results by model id, skipped role -> reason)`.
    """
    out: dict[str, dict[str, object]] = {}
    skipped: dict[str, str] = {}
    for role, (model, url, body, headers, check_listed) in _retrieval_requests(settings, skipped).items():
        entry: dict[str, object] = {"role": role, "available": model in available if check_listed else True}
        started = time.perf_counter()
        try:
            r = client.post(url, json=body, headers=headers)
            ok = r.status_code == 200
            if ok and role == "embedding":
                rows = r.json().get("data") or [{}]
                dim = len(rows[0].get("embedding") or [])
                ok = dim == settings.embedding_dimension
                entry["dimension"] = dim
            entry["reachable"] = ok
            entry["status"] = r.status_code
        except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
            entry["reachable"] = False
            entry["status"] = type(exc).__name__
        entry["latency_ms"] = round((time.perf_counter() - started) * 1000)
        out[model] = entry
    return out, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="send one tiny request per configured model")
    args = parser.parse_args()

    settings = Settings()
    key = settings.llm_key
    if not key:
        print("No LLM key. Set LLM_API_KEY in .env (OPENAI_API_KEY also works for an OpenAI base URL).")
        return 2

    base = settings.nvidia_base_url.rstrip("/")
    with httpx.Client(timeout=60) as client:
        r = client.get(f"{base}/models", headers=_bearer(key))
        if r.status_code != 200:
            print(f"GET /models failed: HTTP {r.status_code} (check the key)")
            return 1
        available = sorted(m["id"] for m in r.json().get("data", []))

        wanted = {
            "generation": settings.generation_models,
            "helper": [settings.helper_model],
            "judge": [settings.judge_model],
        }
        report: dict[str, object] = {
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "base_url": base,
            "available_count": len(available),
            "configured": {},
        }
        configured: dict[str, dict[str, object]] = {}
        for role, models in wanted.items():
            for model in models:
                entry: dict[str, object] = {"role": role, "available": model in available}
                if args.probe and model in available:
                    entry["probe"] = _probe(client, base, model, settings, key)
                configured[model] = entry
        retrieval, skipped = check_retrieval(client, settings, available)
        report["configured"] = configured
        report["retrieval"] = retrieval
        report["skipped"] = skipped
        report["available"] = available

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{len(available)} models visible to this key. Configured models:")
    for model, entry in configured.items():
        mark = "OK " if entry["available"] else "MISSING"
        extra = f"  probe={entry['probe']}" if "probe" in entry else ""
        print(f"  [{mark}] {entry['role']:<10} {model}{extra}")
    print("Hosted retrieval models:")
    if not retrieval:
        print("  (none configured)")
    for model, entry in retrieval.items():
        seen = "listed" if entry["available"] else "not listed"
        mark = "OK " if entry["reachable"] else "FAIL"
        print(f"  [{mark}] {entry['role']:<10} {model}  ({seen}, status={entry['status']})")
    for role, reason in skipped.items():
        print(f"  [SKIP] {role:<10} {reason}")
    missing = [m for m, e in configured.items() if not e["available"]]
    missing += [m for m, e in retrieval.items() if not e["reachable"]]
    if missing:
        print("\nPick replacements from evals/configs/models_available.json and set them in .env.")
    print(f"\nWrote {OUT}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
