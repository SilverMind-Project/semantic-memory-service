from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/semantic_memory"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "semantic-memory-service"
    RETENTION_DAYS: int = 90

    TEXT_EMBEDDING_ENABLED: bool = True
    TRITON_URL: str = "localhost:8701"
    TRITON_TEXT_EMBEDDING_MODEL: str = "embeddinggemma-300m"
    TRITON_TEXT_EMBEDDING_TOKENIZER_PATH: str = "/models/embeddinggemma-300m/1/tokenizer.json"


settings = Settings()
