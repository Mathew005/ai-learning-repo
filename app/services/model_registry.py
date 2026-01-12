"""
Model Registry - Dynamic model discovery and selection.

Discovers available models from Ollama and Gemini at startup,
persists selections to JSON, and provides global model configuration.
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from app.services.ai_providers import AIServiceFactory, ModelInfo


# Config file path
CONFIG_FILE = Path("./data/model_config.json")


class ModelRegistry:
    """
    Central registry for model discovery and selection.
    
    Singleton pattern - call ModelRegistry.instance() to get the registry.
    """
    
    _instance: Optional["ModelRegistry"] = None
    
    def __init__(self):
        self._available_models: List[ModelInfo] = []
        self._config: Dict[str, Any] = {
            "llm_slot_1": None,
            "llm_slot_2": None,
            "embedding": None,
            "last_discovery": None
        }
        self._initialized = False
    
    @classmethod
    def instance(cls) -> "ModelRegistry":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None
    
    # --- Initialization ---
    
    def initialize(self, gemini_api_key: Optional[str] = None, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the registry: load config, discover models, validate selections.
        Call this at application startup.
        """
        if self._initialized:
            return
        
        # Ensure data directory exists
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Load saved config if exists
        self._load_config()
        
        # Discover available models
        self._discover_all_models(gemini_api_key, ollama_base_url)
        
        # Validate and set defaults if needed
        self._validate_and_set_defaults()
        
        # Save config
        self._save_config()
        
        self._initialized = True
    
    def refresh_models(self, gemini_api_key: Optional[str] = None, ollama_base_url: str = "http://localhost:11434"):
        """Re-discover models (called from TUI refresh option)."""
        self._discover_all_models(gemini_api_key, ollama_base_url)
        self._validate_and_set_defaults()
        self._save_config()
    
    # --- Discovery ---
    
    def _discover_all_models(self, gemini_api_key: Optional[str], ollama_base_url: str):
        """Discover models from all available providers."""
        self._available_models = []
        
        # Use Factory to get all configured providers
        # Note: We can pass keys here if needed, but Factory currently pulls from settings.
        
        providers = AIServiceFactory.get_all_services()
        print(f"Discovery: Found {len(providers)} generic services.")

        for provider in providers:
            try:
                models = provider.discover_models()
                self._available_models.extend(models)
            except Exception as e:
                print(f"⚠️ Error discovering models from {provider.provider_name}: {e}")
        
        self._config["last_discovery"] = datetime.now().isoformat()
    
        pass  # Legacy methods removed
    
    # --- Getters ---
    
    def get_available_llms(self) -> List[ModelInfo]:
        """Get all available LLM models."""
        return [m for m in self._available_models if m.type == "llm"]
    
    def get_available_embeddings(self) -> List[ModelInfo]:
        """Get all available embedding models."""
        return [m for m in self._available_models if m.type == "embedding"]
    
    def get_all_models(self) -> List[ModelInfo]:
        """Get all discovered models."""
        return self._available_models.copy()
    
    def get_active_llm(self, slot: int) -> Optional[str]:
        """Get the URN of the active LLM for a slot (1 or 2)."""
        return self._config.get(f"llm_slot_{slot}")
    
    def get_active_embedding(self) -> Optional[str]:
        """Get the URN of the active embedding model."""
        return self._config.get("embedding")
    
    # --- Setters ---
    
    def set_active_llm(self, slot: int, urn: str):
        """Set the active LLM for a slot (1 or 2)."""
        if slot not in (1, 2):
            raise ValueError("Slot must be 1 or 2")
        self._config[f"llm_slot_{slot}"] = urn
        self._save_config()
    
    def set_active_embedding(self, urn: str):
        """Set the active embedding model."""
        self._config["embedding"] = urn
        self._save_config()
    
    # --- Validation & Defaults ---
    
    def _validate_and_set_defaults(self):
        """Ensure selections are valid, set defaults if needed."""
        available_urns = {m.urn for m in self._available_models}
        llms = self.get_available_llms()
        embeddings = self.get_available_embeddings()
        
        # Validate LLM Slot 1
        if not self._config.get("llm_slot_1") or self._config["llm_slot_1"] not in available_urns:
            if llms:
                # Prefer Gemini Flash if available, else first LLM
                flash = next((m for m in llms if "flash" in m.name.lower() and "lite" not in m.name.lower()), None)
                self._config["llm_slot_1"] = flash.urn if flash else llms[0].urn
            else:
                self._config["llm_slot_1"] = None
        
        # Validate LLM Slot 2
        if not self._config.get("llm_slot_2") or self._config["llm_slot_2"] not in available_urns:
            if llms:
                # Prefer Ollama model for slot 2, else second model or same as slot 1
                ollama_llm = next((m for m in llms if m.provider == "ollama"), None)
                if ollama_llm:
                    self._config["llm_slot_2"] = ollama_llm.urn
                elif len(llms) > 1:
                    self._config["llm_slot_2"] = llms[1].urn
                else:
                    self._config["llm_slot_2"] = llms[0].urn
            else:
                self._config["llm_slot_2"] = None
        
        # Validate Embedding
        if not self._config.get("embedding") or self._config["embedding"] not in available_urns:
            if embeddings:
                # Prefer Ollama embedding (local, faster)
                ollama_embed = next((m for m in embeddings if m.provider == "ollama"), None)
                self._config["embedding"] = ollama_embed.urn if ollama_embed else embeddings[0].urn
            else:
                self._config["embedding"] = None
    
    # --- Persistence ---
    
    def _load_config(self):
        """Load config from JSON file."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    saved = json.load(f)
                    self._config.update(saved)
            except Exception as e:
                print(f"⚠️ Could not load model config: {e}")
    
    def _save_config(self):
        """Save config to JSON file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save model config: {e}")
    
    # --- Utility ---
    
    def get_model_info(self, urn: str) -> Optional[ModelInfo]:
        """Get ModelInfo for a specific URN."""
        return next((m for m in self._available_models if m.urn == urn), None)
    
    def has_provider(self, provider: str) -> bool:
        """Check if any models from a provider are available."""
        return any(m.provider == provider for m in self._available_models)
