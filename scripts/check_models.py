"""Check the NVIDIA API key and the configured model ids (feedback item 2).

Usage:  .venv\\Scripts\\python scripts\\check_models.py [--probe]

- Lists every model the key can see (GET /v1/models).
- Reports which of GENERATION_MODELS / HELPER_MODEL / JUDGE_MODEL are missing.
- With --probe, sends one tiny JSON-mode request per configured model and reports
  latency, whether JSON mode was accepted, and whether the reply parsed as JSON.
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


def _probe(client: httpx.Client, base: str, model: str) -> dict[str, object]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": 'Reply with JSON only: {"ok": true, "lang": "<language of the user>"}',
            },
            {"role": "user", "content": "Passport ki fee kitni hai?"},
        ],
        "max_tokens": 60,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    started = time.perf_counter()
    r = client.post(f"{base}/chat/completions", json=body)
    json_mode = True
    if r.status_code in (400, 422):  # model rejects response_format: retry without it
        json_mode = False
        body.pop("response_format")
        r = client.post(f"{base}/chat/completions", json=body)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="send one tiny request per configured model")
    args = parser.parse_args()

    settings = Settings()
    key = settings.nvidia_api_key.get_secret_value() if settings.nvidia_api_key else ""
    if not key:
        print("NVIDIA_API_KEY is not set. Add it to .env (it starts with nvapi-).")
        return 2

    base = settings.nvidia_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    with httpx.Client(headers=headers, timeout=60) as client:
        r = client.get(f"{base}/models")
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
                    entry["probe"] = _probe(client, base, model)
                configured[model] = entry
        report["configured"] = configured
        report["available"] = available

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{len(available)} models visible to this key. Configured models:")
    for model, entry in configured.items():
        mark = "OK " if entry["available"] else "MISSING"
        extra = f"  probe={entry['probe']}" if "probe" in entry else ""
        print(f"  [{mark}] {entry['role']:<10} {model}{extra}")
    missing = [m for m, e in configured.items() if not e["available"]]
    if missing:
        print("\nPick replacements from evals/configs/models_available.json and set them in .env.")
    print(f"\nWrote {OUT}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
