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

    # --- Phase 2: RAG & Embeddings ---
    # Embedding Configuration (Provider/Model)
    # Embedding Configuration (Provider/Model)
    # Define available models here.
    EMBEDDING_MODEL_1: str = "google/models/gemini-embedding-001"
    EMBEDDING_MODEL_2: str = "ollama/embeddinggemma:300m"
    EMBEDDING_MODEL_3: str = "huggingface/all-MiniLM-L6-v2"

    # Active Embedding Provider (Defaults to Model 3 / Local)
    EMBEDDING_PROVIDER_URN: str = "huggingface/all-MiniLM-L6-v2"
    
    # Vector Configuration
    CHROMA_DB_PATH: str = "./data/chroma_db"
    SOURCE_DOCUMENTS_DIR: str = "./data/source_documents"

    # --- Phase 1: LLM Models ---
    # Model Configuration (Standardized Identifiers: "provider/model-name")
    # Examples: "gemini/gemini-flash-latest", "ollama/llama3"
    
    # Generic Model Slot 1 (Primary/Synthesis)
    # MODEL_1: str = "gemini/gemini-flash-latest"
    MODEL_1: str = "ollama/adelnazmy2002/Qwen3-VL-8B-Instruct"
    
    # Generic Model Slot 2 (Analysis/Lite)
    # To use Ollama for analysis: uncomment the line below and comment the gemini line
    # MODEL_2: str = "gemini/gemini-flash-lite-latest"
    MODEL_2: str = "ollama/i82blikeu/gemma-3n-E4B-it-GGUF:Q3_K_M"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()