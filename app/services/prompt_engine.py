from typing import Optional, Dict, Any, Tuple

from app.models.schemas import PromptRequest, AIResponse
from app.core.config import settings
from app.core.exceptions import AIModelError
from app.services.llm_providers import ProviderFactory

# --- Core Execution Helper ---

def _parse_model_urn(model_urn: str) -> Tuple[str, str]:
    """
    Parses a standardized model identifier (URN).
    Format: "provider/model_name"
    Example: "gemini/gemini-1.5-flash" -> ("gemini", "gemini-1.5-flash")
    """
    if "/" not in model_urn:
        raise AIModelError(f"Invalid model identifier format: '{model_urn}'. Expected 'provider/model-name'.")
    
    provider_name, model_alias = model_urn.split("/", 1)
    return provider_name, model_alias

async def _execute_model_call(model_urn: str, request: PromptRequest) -> AIResponse:
    """
    Core helper function to execute a model call.
    model_urn: The standardized identifier string (e.g., 'gemini/gemini-1.5-flash')
    """
    try:
        # 1. Parse the URN to find out WHO to call and WHAT model to ask for
        provider_name, target_model_name = _parse_model_urn(model_urn)
        
        # 2. Get the provider instance
        provider = ProviderFactory.get_provider(provider_name)
        
        # 3. Execute
        return await provider.generate_content(request, target_model_name)

    except AIModelError:
        raise
    except Exception as e:
         raise AIModelError(f"Unexpected error in model execution", {"error": str(e)})


# --- Chain of Thought Steps (Functional Approach) ---

async def _run_analysis_step(request: PromptRequest, model_urn: str) -> AIResponse:
    """
    Step 1: Analyze the problem without solving it.
    """
    analysis_request = PromptRequest(
        user_query=f"Analyze this problem step by step and break it down into logical steps. Do not solve it yet, just analyze: {request.user_query}",
        system_role="You are an analytical sub-agent. Think step by step.",
        temperature=request.temperature
    )
    
    response = await _execute_model_call(model_urn, analysis_request)
    
    if not response.content or len(response.content) < 10:
         raise AIModelError("Analysis model failed to produce valid analysis", {"content": response.content})
         
    return response

async def _run_synthesis_step(request: PromptRequest, analysis_content: str, model_urn: str) -> AIResponse:
    """
    Step 2: Synthesize the final answer based on analysis.
    """
    synthesis_query = f"Original Query: {request.user_query}\n\nAnalysis:\n{analysis_content}\n\nBased on the analysis, provide the final comprehensive answer."
    main_agent_request = PromptRequest(
        user_query=synthesis_query,
        system_role=request.system_role,
        temperature=request.temperature
    )
    
    return await _execute_model_call(model_urn, main_agent_request)


# --- Public API Functions ---

async def call_model_1(request: PromptRequest) -> AIResponse:
    """Call Configured Model 1"""
    return await _execute_model_call(settings.MODEL_1, request)


async def call_model_2(request: PromptRequest) -> AIResponse:
    """Call Configured Model 2"""
    return await _execute_model_call(settings.MODEL_2, request)


async def call_specific_model_by_slot(slot: int, request: PromptRequest) -> AIResponse:
    """Call a specific model slot (1 or 2)"""
    if slot == 1:
        return await call_model_1(request)
    elif slot == 2:
        return await call_model_2(request)
    else:
        raise ValueError("Invalid model slot. Use 1 or 2.")


async def generate_zero_shot_response(request: PromptRequest) -> AIResponse:
    """Generate response using zero-shot approach (defaults to Model 1)"""
    # Kept for backward compatibility if needed, else redundant
    return await call_model_1(request)


async def generate_response(request: PromptRequest) -> AIResponse:
    """Default entry point (defaults to Model 1)"""
    return await call_model_1(request)


async def generate_chain_of_thought_response(request: PromptRequest) -> AIResponse:
    """
    Orchestrates the Chain of Thought sequence.
    Analysis -> Model 2
    Synthesis -> Model 1
    """
    try:
        # Define models for each step (Configurable via settings)
        analysis_urn = settings.MODEL_2
        synthesis_urn = settings.MODEL_1

        # Step 1: Analysis
        analysis_response = await _run_analysis_step(request, analysis_urn)

        # Step 2: Synthesis
        final_response = await _run_synthesis_step(request, analysis_response.content, synthesis_urn)

        # Construct final combined response
        return AIResponse(
            content=final_response.content,
            tokens_used=analysis_response.tokens_used + final_response.tokens_used,
            model_name=f"chain-of-thought",
            steps=[
                {"title": f"Analysis ({analysis_urn})", "content": analysis_response.content, "style": "cyan"},
                {"title": f"Final Answer ({synthesis_urn})", "content": final_response.content, "style": "green"}
            ]
        )
    except AIModelError:
            raise
    except Exception as e:
            raise AIModelError("Chain of Thought failed", {"error": str(e)})
