from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Learning-Repo"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Secrets (Loaded from .env file)
    GEMINI_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()