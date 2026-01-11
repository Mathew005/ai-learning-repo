from fastapi import APIRouter, HTTPException
from app.models.schemas import PromptRequest, AIResponse
from app.services import prompt_engine

router = APIRouter()

@router.post("/ask", response_model=AIResponse, summary="Direct AI Query")
async def run_prompt(payload: PromptRequest):
    """
    Executes a prompt using the specified model slot (Parameter: model_slot).
    Defaults to Model 1.
    """
    try:
        return await prompt_engine.call_specific_model_by_slot(payload.model_slot, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chain-of-thought", response_model=AIResponse, summary="Chain-of-Thought Reasoning")
async def run_chain_of_thought(payload: PromptRequest):
    """
    Executes a Chain-of-Thought reasoning strategy (Model 2 -> Model 1).
    """
    return await prompt_engine.generate_chain_of_thought_response(payload)

# Deprecated/Alias endpoints (optional, keeping clean by removing them as requested)