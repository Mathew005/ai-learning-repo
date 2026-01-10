from pydantic import BaseModel
from typing import Optional, List

# Basic Input for Prompting
class PromptRequest(BaseModel):
    system_role: str = "You are a helpful assistant."
    user_query: str
    temperature: float = 0.7

# Basic Output
class AIResponse(BaseModel):
    content: str
    tokens_used: int
    model_name: str