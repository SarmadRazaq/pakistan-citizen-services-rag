"""Application settings: the only place tunables live (blueprint §12.1).

Every value can be overridden by an environment variable of the same name, or by `.env`.
Secrets are `SecretStr`, so they never appear in reprs, logs or tracebacks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PrivacyMode = Literal["public", "tee_required", "local_only"]
ResponseCacheMode = Literal["off", "eval_only"]
Device = Literal["auto", "cuda", "cpu"]
CsvList = Annotated[list[str], NoDecode]  # "a,b,c" in env, not JSON


def _split_csv(value: object) -> object:
    """Accept `a,b,c` from the environment as well as a JSON list."""
    if isinstance(value, str) and not value.strip().startswith("["):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_question_text: bool = False  # question text is logged at DEBUG only when this is on
    cors_origins: CsvList = Field(default_factory=lambda: ["http://localhost:3000"])
    device: Device = "auto"
    api_prefix: str = "/api/v1"

    # --- LLM endpoints (OpenAI-compatible) ---
    antseed_base_url: str = "http://127.0.0.1:8377/v1"
    antseed_api_key: SecretStr = SecretStr("antseed")  # placeholder: the local proxy needs no key
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_required: bool = False  # readiness does not fail when the local fallback is off
    generation_models: CsvList = Field(
        default_factory=lambda: [
            "deepseek-v4-1-flash",
            "qwen3-235b-a22b-instruct-2507",
            "gemma-4-31b-it",
        ]
    )
    helper_model: str = "deepseek-v4-1-flash"
    judge_model: str = "gpt-oss-120b"
    local_fallback_model: str = "qwen2.5:3b-instruct"
    llm_timeout_s: float = 45.0
    llm_helper_timeout_s: float = 20.0
    llm_max_attempts_per_model: int = 1
    llm_privacy_mode: PrivacyMode = "public"
    eval_pinned_sellers: dict[str, str] = Field(default_factory=dict)

    # --- Local models ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    injection_model: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    embed_batch_size: int = 32
    gpu_fp16: bool = True
    gpu_max_concurrent: int = 1

    # --- Stores ---
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_alias: str = "chunks_active"
    qdrant_timeout_s: float = 5.0
    database_url: str = "sqlite+aiosqlite:///./data/app.sqlite3"
    storage_dir: Path = Path("./storage")

    # --- Retrieval / chunking ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunk_strategy: Literal["recursive", "fixed", "semantic"] = "recursive"
    prefetch_k: int = 30
    candidates_k: int = 20
    context_k: int = 5
    tau_keep: float = 0.2
    tau_refuse: float = 0.3
    dedupe_cosine: float = 0.95
    quote_match_min: int = 90

    # --- Limits ---
    max_upload_mb: int = 50
    sync_ingest_max_mb: int = 10
    max_pages: int = 1000
    max_question_chars: int = 2000
    rate_limit_ask: str = "30/minute"
    rate_limit_upload: str = "10/minute"
    rate_limit_login: str = "5/minute"

    # --- Multi-tenancy (§5.8) ---
    public_owner_id: str = "u_public"
    default_scope: Literal["all", "mine", "public"] = "all"
    max_docs_per_user: int = 50
    max_storage_mb_per_user: int = 200
    max_chunks_per_user: int = 20_000
    max_ingest_jobs_per_user: int = 2
    response_cache: ResponseCacheMode = "eval_only"

    # --- Citizen-services domain (§3.1, GO-9..11) ---
    referrals_file: Path = Path("app/guardrails/referrals.yaml")
    stale_after_months: int = 12
    advice_guard: bool = True
    emergency_guard: bool = True
    jurisdictions: CsvList = Field(default_factory=lambda: ["federal", "punjab", "sindh", "kp"])
    text_quality_min_script_ratio: float = 0.6
    text_quality_min_dict_hit: float = 0.3

    # --- Auth / observability ---
    jwt_secret: SecretStr = SecretStr("change-me-in-.env")
    jwt_ttl_min: int = 30
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None
    langfuse_mask_private_text: bool = True

    # --- Guardrail / ingestion switches ---
    guard_groundedness_check: bool = True
    quarantine_flagged_chunks: bool = False
    ocr_enabled: bool = True
    ocr_langs: str = "eng+urd"

    # --- Health checks ---
    health_probe_timeout_s: float = 2.0

    _split = field_validator("cors_origins", "generation_models", "jurisdictions", mode="before")(_split_csv)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def sync_ingest_max_bytes(self) -> int:
        return self.sync_ingest_max_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings. Tests build their own `Settings(...)` and pass it to `create_app`."""
    return Settings()
