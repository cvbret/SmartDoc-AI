from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str

    redis_host: str = "localhost"
    redis_port: int = 6380
    redis_db: int = 0

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    chroma_collection_name: str = "smartdoc"

    reranker_model: str = "BAAI/bge-reranker-base"

    retrieval_k: int = Field(default=10, gt=0)
    rerank_top_n: int = Field(default=3, gt=0)

    session_expire_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


settings = Settings()
