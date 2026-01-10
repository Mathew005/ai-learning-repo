from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Learning-Repo"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Secrets (Loaded from .env file)
    GEMINI_API_KEY: str

    # Model names
    FLASH_MODEL_NAME: str = "gemini-flash-latest"
    LITE_MODEL_NAME: str = "gemini-flash-lite-latest"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()