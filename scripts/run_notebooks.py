"""Execute every notebook in notebooks/ in place, so the outputs are saved in the .ipynb files.

    .venv/Scripts/python.exe scripts/run_notebooks.py            # all notebooks
    .venv/Scripts/python.exe scripts/run_notebooks.py 03 04      # only those whose name starts with 03 / 04

Runs from the repo root with the interpreter that runs this script (the "python3" kernel is that
interpreter), so `import app` works. With an LLM key set (OPENAI_API_KEY or LLM_API_KEY, environment or .env)
notebooks 03 and 04 use the hosted models; set RAG_NOTEBOOKS_OFFLINE=1 to force the offline fakes. Exits non-zero if any
cell fails.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"


def run(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, kernel_name="python3", timeout=600, resources={"metadata": {"path": str(ROOT)}}
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"FAIL {path.name}\n{exc}")
        return False
    finally:
        nbformat.write(nb, path)
    print(f"ok   {path.name}")
    return True


def main(prefixes: list[str]) -> int:
    if sys.platform == "win32":  # the ZMQ kernel client needs the selector loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    paths = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    if prefixes:
        paths = [p for p in paths if any(p.name.startswith(x) for x in prefixes)]
    if not paths:
        print("no notebooks found")
        return 1
    results = [run(p) for p in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
