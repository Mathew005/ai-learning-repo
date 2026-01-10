from fastapi import APIRouter
from app.models.schemas import PromptRequest, AIResponse
from app.services import prompt_engine

router = APIRouter()

@router.post("/zero-shot", response_model=AIResponse, summary="Zero-Shot Prompting")
async def run_zero_shot(payload: PromptRequest):
    """
    Executes a Zero-Shot prompt strategy using the flash model.
    """
    return await prompt_engine.generate_zero_shot_response(payload)

@router.post("/chain-of-thought", response_model=AIResponse, summary="Chain-of-Thought Reasoning")
async def run_chain_of_thought(payload: PromptRequest):
    """
    Executes a Chain-of-Thought reasoning strategy (Lite -> Flash).
    """
    return await prompt_engine.generate_chain_of_thought_response(payload)

@router.post("/flash-call", response_model=AIResponse, summary="Direct Flash Model Call")
async def run_flash_call(payload: PromptRequest):
    """
    Executes a prompt directly using the flash model.
    """
    return await prompt_engine.call_flash_model(payload)

@router.post("/lite-call", response_model=AIResponse, summary="Direct Lite Model Call")
async def run_lite_call(payload: PromptRequest):
    """
    Executes a prompt directly using the lite model.
    """
    return await prompt_engine.call_lite_model(payload)