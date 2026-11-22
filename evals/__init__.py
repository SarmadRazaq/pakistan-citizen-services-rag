"""Evaluation harness (blueprint §10): golden set, metrics, LLM judge, config runner and reports.

The harness talks to the running service over HTTP only (`/auth/login`, `/search`, `/ask`), so it
measures the real system and never imports `app.generation`, `app.retrieval` or `app.ingestion`.
"""
