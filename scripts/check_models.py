"""Check the LLM key and the configured model ids (feedback item 2).

Usage:  .venv\\Scripts\\python scripts\\check_models.py [--probe]

- Lists every model the LLM key can see (GET {LLM_BASE_URL}/models; the OpenAI API by default).
- Reports which of GENERATION_MODELS / HELPER_MODEL / JUDGE_MODEL are missing.
- With --probe, sends one tiny JSON-mode request per configured model and reports
  latency, whether JSON mode was accepted, and whether the reply parsed as JSON.
- Checks the retrieval models actually configured: the OpenAI embedder at EMBEDDING_DIMENSION
  dimensions (EMBEDDING_PROVIDER=openai) and whether the local reranker can run here (FlagEmbedding
  installed, weights cached). EMBEDDING_PROVIDER=local and RERANKER_PROVIDER=none are not called;
  the output lists what was skipped and why.
- Writes the result to evals/configs/models_available.json (no key inside), so the
  evaluation records exactly which models existed on the day it ran.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


Request = tuple[str, str, dict[str, object], dict[str, str]]  # model, url, body, headers


def _embedding_request(settings: Settings, skipped: dict[str, str]) -> Request | None:
    if settings.embedding_provider != "openai":
        skipped["embedding"] = "EMBEDDING_PROVIDER=local (runs on this machine, nothing to call)"
        return None
    if not settings.openai_key:
        skipped["embedding"] = "OpenAI embedder selected but OPENAI_API_KEY is not set"
        return None
    body: dict[str, object] = {
        "model": settings.openai_embedding_model,
        "input": ["passport"],
        "encoding_format": "float",
        "dimensions": settings.embedding_dimension,
    }
    url = f"{settings.openai_base_url.rstrip('/')}/embeddings"
    return settings.openai_embedding_model, url, body, _bearer(settings.openai_key)


def _hf_cache_dir() -> Path:
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def check_local_reranker(settings: Settings, skipped: dict[str, str]) -> dict[str, object] | None:
    """Can the local reranker run here? Nothing is loaded and nothing is downloaded."""
    if settings.reranker_provider == "none":
        skipped["reranker"] = "RERANKER_PROVIDER=none (retrieval gates on dense similarity)"
        return None
    cached = (_hf_cache_dir() / f"models--{settings.reranker_model.replace('/', '--')}").is_dir()
    return {
        "role": "reranker",
        "package_installed": importlib.util.find_spec("FlagEmbedding") is not None,
        "weights_cached": cached,
    }


def check_retrieval(
    client: httpx.Client, settings: Settings
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Reachability of the OpenAI embeddings and availability of the local reranker.

    Returns `(results by model id, skipped role -> reason)`.
    """
    out: dict[str, dict[str, object]] = {}
    skipped: dict[str, str] = {}
    request = _embedding_request(settings, skipped)
    if request is not None:
        model, url, body, headers = request
        entry: dict[str, object] = {"role": "embedding"}
        started = time.perf_counter()
        try:
            r = client.post(url, json=body, headers=headers)
            ok = r.status_code == 200
            if ok:
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
    reranker = check_local_reranker(settings, skipped)
    if reranker is not None:
        out[settings.reranker_model] = reranker
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

    base = settings.llm_base_url.rstrip("/")
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
        retrieval, skipped = check_retrieval(client, settings)
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
    print("Retrieval models:")
    if not retrieval:
        print("  (none configured)")
    for model, entry in retrieval.items():
        if entry["role"] == "reranker":
            mark = "OK " if entry["package_installed"] else "FAIL"
            cached = "weights cached" if entry["weights_cached"] else "weights download on first use"
            print(f"  [{mark}] {entry['role']:<10} {model}  (local, {cached})")
            continue
        mark = "OK " if entry["reachable"] else "FAIL"
        detail = f"dim={entry.get('dimension')}, status={entry['status']}"
        print(f"  [{mark}] {entry['role']:<10} {model}  ({detail})")
    for role, reason in skipped.items():
        print(f"  [SKIP] {role:<10} {reason}")
    missing = [m for m, e in configured.items() if not e["available"]]
    missing += [m for m, e in retrieval.items() if not e.get("reachable", e.get("package_installed"))]
    if missing:
        print("\nPick replacements from evals/configs/models_available.json and set them in .env.")
    print(f"\nWrote {OUT}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
