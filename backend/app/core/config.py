from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate .env in backend directory or project root
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:142@localhost:5432/northstar"
    EMBEDDING_SYNC_ENABLED: bool = True
    EMBEDDING_SYNC_INTERVAL_SECONDS: int = 300
    EMBEDDING_BATCH_SIZE: int = 64

    model_config = SettingsConfigDict(
        env_file=(str(ENV_FILE), ".env"),
        extra="ignore",
    )


settings = Settings()