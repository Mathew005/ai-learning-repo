import os
from typing import List, Generator, Any
from dotenv import load_dotenv
from litellm import completion, embedding

# Auto-load .env from PWD (root)
load_dotenv()
from shared.settings import settings

class LLMEngine:
    """
    Centralized LLM Handler using Configured Slots.
    """
    @staticmethod
    def get_model_slots():
        """
        Returns the configured model slots from settings.
        Dynamically finds all settings starting with MODEL_
        """
        return {
            k: v for k, v in settings.model_dump().items() 
            if k.startswith("MODEL_") and v
        }

    @staticmethod
    def get_embedding_slots():
        """
        Returns the configured embedding slots from settings.
        Dynamically finds all settings starting with EMBED_
        """
        return {
            k: v for k, v in settings.model_dump().items() 
            if k.startswith("EMBED_") and v
        }

    @staticmethod
    def embed(model_name: str, input: Any) -> List[float]:
        """
        Universal embedding function.
        """
        try:
            kwargs = {}
            if model_name.startswith("ollama/"):
                kwargs["api_base"] = settings.OLLAMA_BASE_URL

            response = embedding(model=model_name, input=input, **kwargs)
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Embedding Error: {e}")
            return []

    @staticmethod
    def chat(model_name: str, messages: List[dict], temperature: float = 0.7, stream: bool = False) -> Any:
        """
        Universal chat function.
        If stream=False (default): returns just the content string.
        If stream=True: returns a generator for streaming.
        """
        try:
            kwargs = {}
            if model_name.startswith("ollama/"):
                kwargs["api_base"] = settings.OLLAMA_BASE_URL

            response = completion(
                model=model_name,
                messages=messages,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
        except Exception as e:
            # Return error as a string for UI handling
            return f"Error: {str(e)}"

# Singleton instance if needed, or just use static methods
engine = LLMEngine()
