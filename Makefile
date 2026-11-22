# Linux/macOS/CI shortcuts. On Windows use scripts/dev.ps1 (same targets).
PY := .venv/bin/python

.PHONY: install pin lint fmt types test check run qdrant ready corpus eval

install:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -r requirements.txt -r requirements-dev.txt

pin:
	uv pip compile --universal requirements.in -o requirements.txt
	uv pip compile --universal requirements-dev.in -o requirements-dev.txt

lint:
	$(PY) -m ruff check app tests
	$(PY) -m ruff format --check app tests

fmt:
	$(PY) -m ruff check app tests --fix
	$(PY) -m ruff format app tests

types:
	$(PY) -m mypy

test:
	$(PY) -m pytest -m "not live"

check: lint types test

run:
	$(PY) -m uvicorn app.main:app_factory --factory --reload --port 8000

qdrant:
	docker compose up -d qdrant

ready:
	curl -s http://127.0.0.1:8000/health/ready

corpus:
	$(PY) scripts/collect_corpus.py

eval:
	$(PY) -m evals.run
