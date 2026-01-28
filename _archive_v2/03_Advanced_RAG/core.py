from abc import ABC, abstractmethod
from typing import Any
from rich.console import Console
from app.core.config import settings

# Common console for experiments to use
console = Console()

class LabExperiment(ABC):
    """
    Abstract Base Class for all LangChain Lab Experiments.
    Enforces a consistent interface for the TUI to execute.
    """
    
    def name(self) -> str:
        """The display name of the experiment (e.g. 'Simple Chain')."""
        raise NotImplementedError

    def category(self) -> str:
        """The category/topic this belongs to (e.g. 'Topic 1: LCEL')."""
        raise NotImplementedError

    @abstractmethod
    async def run(self) -> Any:
        """
        The main execution logic. 
        Experiments should print their own output to the console 
        or return a result that the TUI can print.
        """
        pass

    def get_model_config(self, provider: str = "google"):
        """
        Helper to get standardized model params based on our config.py.
        Attempts to parse model name from MODEL_1/MODEL_2 settings.
        
        Example URNs in config: "google/gemini-1.5-flash", "ollama/gemma:2b"
        """
        # Determine which model to use. For labs, we default to MODEL_1 (primary).
        # You could also add logic to choose MODEL_2 if requested.
        urn = settings.MODEL_1 
        
        # Simple parser: defined as "provider/model_name"
        try:
            config_provider, config_model = urn.split("/", 1)
        except ValueError:
             # Fallback if config is malformed
             config_provider = "google"
             config_model = "gemini-flash-latest"

        if provider == "google":
            # If the config URN matches the requested provider, use the config model.
            # Otherwise, use a safe default or warn.
            model_name = config_model if config_provider == "google" else "gemini-flash-latest"
            return {
                "model": model_name,
                "google_api_key": settings.GEMINI_API_KEY,
                "temperature": 0
            }
        elif provider == "ollama":
            model_name = config_model if config_provider == "ollama" else "gemma:2b"
            # Handle potential sub-paths in model name if needed, but usually just passing it works.
            return {
                "model": model_name,
                "base_url": settings.OLLAMA_BASE_URL,
                "temperature": 0
            }
        return {}
