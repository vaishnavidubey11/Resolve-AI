from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./resolve.db"
    jwt_secret: str = "dev-only-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    customer_token_expire_minutes: int = 720

    llm_provider: str = "mock"
    embedding_provider: str = "mock"
    embedding_dim: int = 1024
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    provider_timeout_seconds: float = 20.0
    provider_max_retries: int = 2

    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 4
    retrieval_min_score: float | None = None  # None = provider-recommended default
    history_window: int = 6

    max_upload_bytes: int = 5 * 1024 * 1024
    max_message_chars: int = 2000

    rate_limit_enabled: bool = True
    login_rate_limit_per_min: int = 10
    message_rate_limit_per_min: int = 30

    order_api_mode: str = "mock"
    order_api_base_url: str = ""
    order_api_key: str = ""

    conversation_retention_days: int = Field(default=0, ge=0)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
