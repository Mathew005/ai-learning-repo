from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Learning-Repo"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Secrets (Loaded from .env file)
    GEMINI_API_KEY: str

    # External Service Configs
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Model Configuration (Standardized Identifiers: "provider/model-name")
    # Examples: "gemini/gemini-flash-latest", "ollama/llama3"
    
    # Generic Model Slot 1 (Primary/Synthesis)
    MODEL_1: str = "gemini/gemini-flash-latest"
    
    # Generic Model Slot 2 (Analysis/Lite)
    # To use Ollama for analysis: uncomment the line below and comment the gemini line
    MODEL_2: str = "ollama/i82blikeu/gemma-3n-E4B-it-GGUF:Q3_K_M"
    # MODEL_2: str = "gemini/gemini-flash-lite-latest"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()