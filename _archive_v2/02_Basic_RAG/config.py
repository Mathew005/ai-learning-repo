"""
Configuration module for CoCo RAG application.

Loads settings from .env file and provides model configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent / ".env")


class Models:
    """Available model configurations."""
    
    # Ollama models (local)
    OLLAMA_GEMMA = "ollama/gemma:2b"
    
    # Google models (API)
    GOOGLE_GEMINI = "google/gemini-flash-latest"


class Config:
    """Application configuration loaded from environment."""
    
    # LLM Provider: "ollama" or "google"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    
    # Active model based on provider
    @classmethod
    def get_model(cls) -> str:
        if cls.LLM_PROVIDER == "google":
            return Models.GOOGLE_GEMINI
        return Models.OLLAMA_GEMMA
    
    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Google settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Embedding settings (always Ollama)
    EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # Retrieval settings
    TOP_K: int = 3
    
    # ChromaDB settings
    CHROMA_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "coco_collection"
