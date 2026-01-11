from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import httpx
import json
from google import genai
from google.genai import types as genai_types
from app.models.schemas import PromptRequest, AIResponse
from app.core.config import settings
from app.core.exceptions import AIModelError

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_content(self, request: PromptRequest, model_name: str) -> AIResponse:
        """
        Generates content from the LLM.
        """
        pass

class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini models."""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def generate_content(self, request: PromptRequest, model_name: str) -> AIResponse:
        try:
            config = genai_types.GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=2048,
                system_instruction=request.system_role
            )

            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=request.user_query,
                    config=config
                )
            except Exception as e:
                # Wrap SDK errors
                raise AIModelError(f"Failed to generate content from Gemini ({model_name})", {"original_error": str(e)})

            tokens_used = 0
            if response.usage_metadata:
                tokens_used = response.usage_metadata.total_token_count

            return AIResponse(
                content=response.text,
                tokens_used=tokens_used,
                model_name=model_name
            )

        except AIModelError:
            raise
        except Exception as e:
             raise AIModelError(f"Unexpected error in Gemini service", {"error": str(e)})

class OllamaProvider(LLMProvider):
    """Provider for Ollama models (via raw HTTP API)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    async def generate_content(self, request: PromptRequest, model_name: str) -> AIResponse:
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": request.system_role},
                {"role": "user", "content": request.user_query}
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(url, json=payload, timeout=60.0)
                    response.raise_for_status()
                    data = response.json()
                except httpx.RequestError as e:
                     raise AIModelError(f"Ollama connection failed", {"error": str(e)})
                except httpx.HTTPStatusError as e:
                     raise AIModelError(f"Ollama returned error status", {"status": e.response.status_code, "response": e.response.text})

            content = data.get("message", {}).get("content", "")
            # Ollama returns token counts in various fields, total_duration etc.
            # prompt_eval_count + eval_count is roughly total tokens
            tokens_used = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)

            return AIResponse(
                content=content,
                tokens_used=tokens_used,
                model_name=model_name
            )
        
        except AIModelError:
            raise
        except Exception as e:
            raise AIModelError(f"Unexpected error in Ollama service", {"error": str(e)})

class ProviderFactory:
    """Factory to get the correct LLM provider."""

    _instances: Dict[str, LLMProvider] = {}

    @classmethod
    def get_provider(cls, provider_name: str) -> LLMProvider:
        if provider_name in cls._instances:
            return cls._instances[provider_name]

        if provider_name == "gemini":
            instance = GeminiProvider(api_key=settings.GEMINI_API_KEY)
        elif provider_name == "ollama":
            # Strip /v1 if it was added for OpenAI compatibility, as we are now using native API
            base_url = settings.OLLAMA_BASE_URL.replace("/v1", "")
            instance = OllamaProvider(base_url=base_url)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        
        cls._instances[provider_name] = instance
        return instance
