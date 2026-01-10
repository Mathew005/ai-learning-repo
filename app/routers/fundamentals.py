from fastapi import APIRouter
from app.models.schemas import PromptRequest, AIResponse
from app.services import prompt_engine

router = APIRouter()

@router.post("/zero-shot", response_model=AIResponse)
async def run_zero_shot(payload: PromptRequest):
    """
    Executes a Zero-Shot prompt strategy.
    """
    result = await prompt_engine.generate_response(payload)
    return result

@router.post("/chain-of-thought", response_model=AIResponse)
async def run_chain_of_thought(payload: PromptRequest):
    """
    Executes a Chain-of-Thought reasoning strategy.
    """
    # In the future, you inject specific CoT logic here
    result = await prompt_engine.generate_response(payload)
    return result