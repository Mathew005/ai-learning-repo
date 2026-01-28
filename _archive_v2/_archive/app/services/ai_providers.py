import httpx
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from google import genai
from google.genai import types as genai_types
from app.core.config import settings
from app.core.exceptions import AIModelError
from app.models.schemas import PromptRequest, AIResponse

# --- Data Structures ---

@dataclass
class ModelInfo:
    """Information about a discovered model."""
    urn: str            # "ollama/gemma:2b"
    provider: str       # "ollama"
    name: str           # "gemma:2b"
    type: str           # "llm" or "embedding"
    size_gb: Optional[float] = None
    params: Optional[str] = None

def parse_model_urn(urn: str) -> Tuple[str, str]:
    """
    Parses a standardized model identifier (URN).
    Format: "provider/model_name"
    """
    if "/" not in urn:
        raise ValueError(f"Invalid model identifier format: '{urn}'. Expected 'provider/model-name'.")
    provider, model = urn.split("/", 1)
    return provider.lower(), model

# --- Interfaces ---

class IGenerative(ABC):
    @abstractmethod
    async def generate_content(self, request: PromptRequest, model_name: str) -> AIResponse:
        pass

class IEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str, model_name: str) -> List[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], model_name: str) -> List[List[float]]:
        pass

class AIProvider(IGenerative, IEmbedder):
    """
    Unified abstract base class for an AI Provider.
    Must implement Discovery, Generation, and Embedding capabilities.
    """
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def discover_models(self) -> List[ModelInfo]:
        pass


# --- Implementations ---

class GeminiService(AIProvider):
    def __init__(self, api_key: str):
        if not api_key:
             # We allow init without key, but methods might fail or discovery returns empty
             self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    # -- Discovery --
    def discover_models(self) -> List[ModelInfo]:
        models = []
        if not self.client:
            return models
        try:
            for m in self.client.models.list(config={'query_base': True}):
                name_lower = m.name.lower()
                actions = m.supported_actions or []
                is_embedding = any("embed" in a for a in actions)

                if is_embedding or "latest" in name_lower:
                    model_type = None
                    if is_embedding:
                        if "exp" not in name_lower and "gecko" not in name_lower:
                            model_type = "embedding"
                    elif 'generateContent' in actions:
                        model_type = "llm"
                    
                    if model_type:
                        model_name = m.name.replace("models/", "")
                        models.append(ModelInfo(
                            urn=f"gemini/{model_name}",
                            provider="gemini",
                            name=model_name,
                            type=model_type
                        ))
        except Exception as e:
            print(f"⚠️ Gemini Discovery Error: {e}")
        return models

    # -- Generation --
    async def generate_content(self, request: PromptRequest, model_name: str) -> AIResponse:
        if not self.client:
             raise AIModelError("Gemini API Key not configured.")
        
        try:
            config = genai_types.GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=2048,
                system_instruction=request.system_role
            )
            # Prepare contents with history if available
            contents = []
            if request.history:
                for msg in request.history:
                    role = "model" if msg["role"] in ["model", "assistant"] else "user"
                    contents.append(genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=msg["content"])]))
            
            # Add current query
            contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=request.user_query)]))

            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            tokens_used = response.usage_metadata.total_token_count if response.usage_metadata else 0
            
            return AIResponse(
                content=response.text,
                tokens_used=tokens_used,
                model_name=model_name
            )
        except Exception as e:
            raise AIModelError(f"Gemini Generation Error ({model_name}): {e}")

    # -- Embedding --
    def embed_text(self, text: str, model_name: str) -> List[float]:
        return self.embed_batch([text], model_name)[0]

    def embed_batch(self, texts: List[str], model_name: str) -> List[List[float]]:
        if not self.client:
            raise AIModelError("Gemini API Key not configured.")
        try:
            response = self.client.models.embed_content(
                model=model_name,
                contents=texts
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            raise AIModelError(f"Gemini Embedding Error ({model_name}): {e}")


class OllamaService(AIProvider):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    @property
    def provider_name(self) -> str:
        return "ollama"

    # -- Discovery --
    def discover_models(self) -> List[ModelInfo]:
        models = []
        try:
            import ollama
            # Assuming env vars or default config for the library
            response = ollama.list()
            for m in response.models:
                families = [f.lower() for f in (getattr(m.details, 'families', []) or [])]
                is_embedding = 'embedding' in families or 'embed' in m.model.lower()
                models.append(ModelInfo(
                    urn=f"ollama/{m.model}",
                    provider="ollama",
                    name=m.model,
                    type="embedding" if is_embedding else "llm",
                    size_gb=round(m.size / (1024**3), 2),
                    params=getattr(m.details, 'parameter_size', None)
                ))
        except Exception as e:
            print(f"⚠️ Ollama Discovery Error: {e}")
        return models

    # -- Generation --
    async def generate_content(self, request: PromptRequest, model_name: str) -> AIResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model_name,
            "model": model_name,
            "messages": [
                {"role": "system", "content": request.system_role},
            ],
            "stream": False,
            "options": {"temperature": request.temperature}
        }
        
        # Add history if available
        if request.history:
            for msg in request.history:
                # Ollama uses 'assistant' for model responses
                role = "assistant" if msg["role"] in ["model", "assistant"] else "user"
                payload["messages"].append({"role": role, "content": msg["content"]})
        
        # Add current user query
        payload["messages"].append({"role": "user", "content": request.user_query})
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
            
            content = data.get("message", {}).get("content", "")
            tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            return AIResponse(content=content, tokens_used=tokens, model_name=model_name)
        except Exception as e:
            raise AIModelError(f"Ollama Generation Error: {e}")

    # -- Embedding --
    def embed_text(self, text: str, model_name: str) -> List[float]:
        # Ollama python lib is sync, or use httpx for async/sync mixed. 
        # Interface is sync here for now to match legacy Validation.
        # But wait, original code used httpx post.
        try:
            # We can use httpx.post synchronously
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model_name, "prompt": text},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise AIModelError(f"Ollama Embedding Error: {e}")

    def embed_batch(self, texts: List[str], model_name: str) -> List[List[float]]:
        return [self.embed_text(t, model_name) for t in texts]


# --- Factory ---

class AIServiceFactory:
    _instances: Dict[str, AIProvider] = {}

    @classmethod
    def get_service(cls, provider_name: str) -> AIProvider:
        if provider_name in cls._instances:
            return cls._instances[provider_name]
        
        instance = None
        if provider_name == "gemini":
            instance = GeminiService(settings.GEMINI_API_KEY)
        elif provider_name == "ollama":
            base = settings.OLLAMA_BASE_URL.replace("/v1", "")
            instance = OllamaService(base)
        else:
            raise ValueError(f"Unknown Provider: {provider_name}")
        
        cls._instances[provider_name] = instance
        return instance

    @classmethod
    def get_all_services(cls) -> List[AIProvider]:
        services = []
        # Always try to register known ones
        services.append(cls.get_service("gemini"))
        services.append(cls.get_service("ollama"))
        return services
