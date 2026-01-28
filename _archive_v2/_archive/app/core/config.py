from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Learning-Repo"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Secrets (Loaded from .env file)
    GEMINI_API_KEY: Optional[str] = None

    # External Service Configs
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Vector Configuration
    CHROMA_DB_PATH: str = "./data/chroma_db"
    SOURCE_DOCUMENTS_DIR: str = "./data/source_documents"
    
    # Model Configuration - NOW DYNAMIC
    # Models are discovered at runtime via ModelRegistry.
    # Selections are persisted in data/model_config.json
    # Use: from app.services.model_registry import ModelRegistry
    #      registry = ModelRegistry.instance()
    #      llm_1 = registry.get_active_llm(1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )


settings = Settings()