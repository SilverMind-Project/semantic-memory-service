import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/semantic_memory")
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "semantic-memory-service"
    
    # Retention
    RETENTION_DAYS: int = 90

    class Config:
        env_file = ".env"

settings = Settings()
