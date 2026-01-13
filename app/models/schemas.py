from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict

# Basic Input for Prompting
class PromptRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    system_role: str = "You are a helpful assistant."
    user_query: str
    temperature: float = 0.7
    model_slot: int = 1  # Default to Model 1
    history: Optional[List[Dict[str, str]]] = None  # Conversation history [{"role": "user"|"model", "content": "..."}]

# Basic Output
class AIResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    content: str
    tokens_used: int
    model_name: str
    steps: Optional[List[Dict[str, str]]] = None