"""Application settings: the only place tunables live (blueprint §12.1).

Every value can be overridden by an environment variable of the same name, or by `.env`.
Secrets are `SecretStr`, so they never appear in reprs, logs or tracebacks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PrivateContextPolicy = Literal["hosted_with_notice", "local_only"]
ResponseCacheMode = Literal["off", "eval_only"]
Device = Literal["auto", "cuda", "cpu"]
Provider = Literal["nvidia", "local"]
EmbeddingProvider = Literal["nvidia", "local", "openai"]
RerankerProvider = Literal["nvidia", "local", "none"]
LlmTokenParam = Literal["max_tokens", "max_completion_tokens"]
InjectionProvider = Literal["nvidia", "none"]
DEFAULT_JWT_SECRET = "change-me-in-.env"  # noqa: S105 - the placeholder, rejected in prod
MIN_JWT_SECRET_CHARS = 32
CsvList = Annotated[list[str], NoDecode]  # "a,b,c" in env, not JSON


def _split_csv(value: object) -> object:
    """Accept `a,b,c` from the environment as well as a JSON list."""
    if isinstance(value, str) and not value.strip().startswith("["):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        populate_by_name=True,  # fields with an env alias can still be set by field name (tests, code)
    )

    # --- App ---
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_question_text: bool = False  # question text is logged at DEBUG only when this is on
    cors_origins: CsvList = Field(default_factory=lambda: ["http://localhost:3000"])
    device: Device = "auto"
    api_prefix: str = "/api/v1"

    # --- LLM endpoints (OpenAI-compatible) ---
    # Hosted: any OpenAI-compatible endpoint. Default NVIDIA API catalog (build.nvidia.com); set
    # LLM_BASE_URL=https://api.openai.com/v1 for OpenAI. The Python fields keep their historical names
    # (`nvidia_*`) but read the neutral env names first: LLM_BASE_URL / LLM_API_KEY, then the old
    # NVIDIA_BASE_URL / NVIDIA_API_KEY. Keys go in .env, never in code.
    # Model ids are verified against GET /v1/models by scripts/check_models.py.
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "NVIDIA_BASE_URL", "nvidia_base_url"),
    )
    nvidia_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LLM_API_KEY", "NVIDIA_API_KEY", "nvidia_api_key")
    )
    # Key for the NVIDIA retrieval endpoints (embeddings / reranker / jailbreak) when it differs from the
    # LLM key, e.g. LLM_API_KEY is an OpenAI key. Falls back to the LLM key while the LLM base URL is NVIDIA.
    nvidia_retrieval_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "NVIDIA_RETRIEVAL_API_KEY", "NVIDIA_API_KEY", "nvidia_retrieval_api_key"
        ),
    )
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-large"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_required: bool = False  # readiness does not fail when the local fallback is off
    generation_models: CsvList = Field(
        default_factory=lambda: [
            "deepseek-ai/deepseek-v3.1",
            "qwen/qwen3-235b-a22b",
            "meta/llama-3.3-70b-instruct",
        ]
    )
    helper_model: str = "meta/llama-3.3-70b-instruct"
    judge_model: str = "openai/gpt-oss-120b"
    local_fallback_model: str = "qwen2.5:3b-instruct"
    llm_timeout_s: float = 45.0
    llm_helper_timeout_s: float = 20.0
    llm_max_attempts_per_model: int = 1
    llm_max_rpm: int = 35  # stay under the NVIDIA free-tier rate limit (~40 requests/minute)
    llm_json_mode: bool = True  # send response_format={"type":"json_object"} where the model accepts it
    # OpenAI reasoning models (gpt-5*, o*) reject `max_tokens` and a non-default `temperature`: set
    # LLM_TOKEN_PARAM=max_completion_tokens to send `max_completion_tokens` and omit `temperature`.
    llm_token_param: LlmTokenParam = "max_tokens"  # noqa: S105 - a request-field name, not a secret
    # Where an answer that uses PRIVATE uploads may be generated (feedback item 5):
    #   hosted_with_notice: NVIDIA API, and the answer carries a third_party_processing notice
    #   local_only: private-context answers go to the local Ollama model only
    private_context_policy: PrivateContextPolicy = "hosted_with_notice"
    llm_cache_dir: Path = Path("evals/.cache/llm")  # eval response cache (RESPONSE_CACHE=eval_only)
    # Cost accounting (B3). MODEL_PRICES is JSON: {"model-id": [usd_per_1M_input, usd_per_1M_output]}.
    # Empty by default: no prices are invented. LLM_BILLED (old name NVIDIA_BILLED) says whether hosted calls
    # cost real money; unset = billed when the LLM base URL is OpenAI's, free otherwise (NVIDIA free tier).
    model_prices: dict[str, tuple[float, float]] = Field(default_factory=dict)
    llm_billed: bool | None = Field(
        default=None, validation_alias=AliasChoices("LLM_BILLED", "NVIDIA_BILLED", "llm_billed")
    )

    # --- Hosted retrieval models (default; NVIDIA API catalog, same NVIDIA_API_KEY) ---
    embedding_provider: EmbeddingProvider = "nvidia"  # nvidia | local | openai
    hosted_embedding_model: str = "baai/bge-m3"
    embedding_dimension: int = 1024
    reranker_provider: RerankerProvider = "nvidia"  # nvidia | local | none (gate on dense similarity)
    hosted_reranker_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    hosted_reranker_url: str | None = None  # default derived from nvidia_retrieval_base_url + model
    nvidia_retrieval_base_url: str = "https://ai.api.nvidia.com/v1/retrieval"
    injection_provider: InjectionProvider = "nvidia"
    jailbreak_detect_url: str = "https://ai.api.nvidia.com/v1/security/nvidia/nemoguard-jailbreak-detect"
    hosted_retrieval_timeout_s: float = 20.0
    # Ollama answers when every hosted model failed (degraded=True). Off by default; required by
    # private_context_policy=local_only.
    local_fallback_enabled: bool = False

    # --- Local models (optional: EMBEDDING_PROVIDER / RERANKER_PROVIDER = local, or the tokenizer) ---
    embedding_model: str = "BAAI/bge-m3"  # also the tokenizer that sizes Chunks
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
    sync_ingest_max_pages: int = 80
    background_ingest_workers: int = Field(default=2, ge=1)
    background_ingest_queue_max: int = Field(default=20, ge=1)
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
    # Honour the X-Eval-Config header (admin callers only) so the eval harness can compare configurations.
    # Off by default and never needed in production.
    eval_overrides_enabled: bool = False

    # --- Citizen-services domain (§3.1, GO-9..11) ---
    referrals_file: Path = Path("app/guardrails/referrals.yaml")
    stale_after_months: int = 12
    advice_guard: bool = True
    emergency_guard: bool = True
    jurisdictions: CsvList = Field(default_factory=lambda: ["federal", "punjab", "sindh", "kp"])
    text_quality_min_script_ratio: float = 0.6
    text_quality_min_dict_hit: float = 0.15  # measured: real EN table page 0.25, scrambled Urdu 0.01-0.03

    # --- Auth / observability ---
    jwt_secret: SecretStr = SecretStr(DEFAULT_JWT_SECRET)
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

    @model_validator(mode="after")
    def _local_only_needs_local_model(self) -> Settings:
        if self.private_context_policy == "local_only" and not self.local_fallback_enabled:
            raise ValueError("PRIVATE_CONTEXT_POLICY=local_only requires LOCAL_FALLBACK_ENABLED=true")
        return self

    @model_validator(mode="after")
    def _production_checks(self) -> Settings:
        """Refuse to start in prod with insecure defaults. Lists every problem; never echoes a secret."""
        if self.app_env != "prod":
            return self
        problems: list[str] = []
        secret = self.jwt_secret.get_secret_value()
        if secret == DEFAULT_JWT_SECRET:
            problems.append("JWT_SECRET is still the default value")
        elif len(secret) < MIN_JWT_SECRET_CHARS:
            problems.append(f"JWT_SECRET must be at least {MIN_JWT_SECRET_CHARS} characters")
        if not self.llm_key:
            problems.append(
                "LLM_API_KEY (alias NVIDIA_API_KEY, or OPENAI_API_KEY on an OpenAI base URL) must be set"
            )
        if self.embedding_provider == "openai" and not self.openai_key:
            problems.append("OPENAI_API_KEY must be set (EMBEDDING_PROVIDER=openai)")
        if (
            self.needs_nvidia_retrieval_key
            and not self.nvidia_key
            and not (self.llm_is_nvidia and not self.llm_key)
        ):
            problems.append("NVIDIA_API_KEY must be set (an NVIDIA retrieval provider is selected)")
        if not self.cors_origins:
            problems.append("CORS_ORIGINS must not be empty")
        elif "*" in self.cors_origins:
            problems.append("CORS_ORIGINS must not contain '*'")
        if self.log_question_text:
            problems.append("LOG_QUESTION_TEXT must be false")
        if problems:
            raise ValueError("Invalid production configuration: " + "; ".join(problems))
        return self

    @staticmethod
    def _host(url: str) -> str:
        return (urlparse(url).hostname or "").lower()

    @property
    def llm_is_openai(self) -> bool:
        return self._host(self.nvidia_base_url) == "api.openai.com"

    @property
    def llm_is_nvidia(self) -> bool:
        return self._host(self.nvidia_base_url).endswith("nvidia.com")

    @staticmethod
    def _secret(value: SecretStr | None) -> str:
        return value.get_secret_value().strip() if value is not None else ""

    @property
    def llm_key(self) -> str:
        """Chat key: LLM_API_KEY / NVIDIA_API_KEY, else OPENAI_API_KEY on an OpenAI base URL."""
        key = self._secret(self.nvidia_api_key)
        if not key and self.llm_is_openai:
            key = self._secret(self.openai_api_key)
        return key

    @property
    def openai_key(self) -> str:
        """Key for OpenAI embeddings: OPENAI_API_KEY, else the LLM key when the LLM base URL is OpenAI's."""
        key = self._secret(self.openai_api_key)
        if not key and self.llm_is_openai:
            key = self._secret(self.nvidia_api_key)
        return key

    @property
    def nvidia_key(self) -> str:
        """Key for the NVIDIA retrieval endpoints. The LLM key is only reused while the LLM base URL is
        NVIDIA's, so an OpenAI key is never sent to NVIDIA."""
        key = self._secret(self.nvidia_retrieval_api_key)
        if not key and self.llm_is_nvidia:
            key = self._secret(self.nvidia_api_key)
        return key

    @property
    def needs_nvidia_retrieval_key(self) -> bool:
        return "nvidia" in (self.embedding_provider, self.reranker_provider, self.injection_provider)

    @property
    def hosted_billed(self) -> bool:
        """Do hosted LLM calls cost real money? LLM_BILLED if set, else true for OpenAI, false otherwise."""
        return self.llm_billed if self.llm_billed is not None else self.llm_is_openai

    @property
    def reranker_url(self) -> str:
        """Reranking endpoint: HOSTED_RERANKER_URL, else derived from the retrieval base URL and model."""
        if self.hosted_reranker_url:
            return self.hosted_reranker_url
        model = self.hosted_reranker_model.replace(".", "_")
        return f"{self.nvidia_retrieval_base_url.rstrip('/')}/{model}/reranking"

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
