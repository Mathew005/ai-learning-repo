from abc import ABC, abstractmethod
from typing import List, Optional
import httpx
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.exceptions import AIModelError
import contextlib

# --- Abstract Base Class ---
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of strings."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider specific name for collection mapping."""
        pass


# --- Google / Gemini Provider ---
class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
             raise ValueError("Google API Key is required for GoogleEmbeddingProvider")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @property
    def provider_name(self) -> str:
        return "google"

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            # Response has 'embeddings' which is a list of Embedding objects
            return [e.values for e in response.embeddings]
        except Exception as e:
            raise AIModelError(f"Google Embedding Error ({self.model_name}): {str(e)}")


# --- Ollama Provider ---
class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    @property
    def provider_name(self) -> str:
        return "ollama"

    def embed_text(self, text: str) -> List[float]:
        try:
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            msg = str(e)
            if "connection" in msg.lower() or "refused" in msg.lower():
                 msg += " (Is 'ollama serve' running?)"
            raise AIModelError(f"Ollama Embedding Error ({self.model_name}): {msg}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Ollama API doesn't strictly support batch in one call usually, loop it.
        return [self.embed_text(t) for t in texts]


# --- HuggingFace / Sentence Transformers Provider ---
class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Lazy load to avoid startup cost if not used
        print(f"Loading Sentence Transformer Model '{model_name}' (this may take a moment)...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


# --- Factory ---
class EmbeddingFactory:
    _instance: Optional[EmbeddingProvider] = None
    _current_urn: Optional[str] = None

    @classmethod
    def get_provider(cls) -> EmbeddingProvider:
        # Import here to avoid circular imports
        from app.services.model_registry import ModelRegistry
        
        registry = ModelRegistry.instance()
        urn = registry.get_active_embedding()
        
        # Return cached instance if URN hasn't changed
        if cls._instance and cls._current_urn == urn:
            return cls._instance
        
        if not urn:
            raise ValueError("No embedding model configured. Run model discovery first.")
        
        if "/" not in urn:
            raise ValueError(f"Invalid embedding URN format: {urn}. Expected 'provider/model-name'.")

        provider_type, model_name = urn.split("/", 1)
        provider_type = provider_type.lower()
        
        if provider_type == "gemini" or provider_type == "google":
            cls._instance = GoogleEmbeddingProvider(
                api_key=settings.GEMINI_API_KEY,
                model_name=model_name
            )
        elif provider_type == "ollama":
            cls._instance = OllamaEmbeddingProvider(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=model_name
            )
        elif provider_type == "huggingface":
            cls._instance = HuggingFaceEmbeddingProvider(
                model_name=model_name
            )
        else:
            raise ValueError(f"Unknown embedding provider type: {provider_type} (URN: {urn})")
        
        cls._current_urn = urn
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the cached instance (call when embedding selection changes)."""
        cls._instance = None
        cls._current_urn = None
