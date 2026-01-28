from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Model Slots - 'Loadouts' for the application
    MODEL_1: str = "gemini/gemini-flash-latest"
    MODEL_2: str = "ollama/gemma:2b"
    MODEL_3: str = "ollama/gemma3n"

    # Embedding Slots
    EMBED_1: str = "gemini/text-embedding-004"
    EMBED_2: str = "ollama/nomic-embed-text"
    EMBED_3: str = "ollama/embeddinggemma"

    class Config:
        env_file = ".env"
        extra = "ignore" # Ignore other keys in .env

# Singleton Instance
settings = Settings()
