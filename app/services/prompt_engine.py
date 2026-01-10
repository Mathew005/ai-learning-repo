from typing import Optional, Dict, Any
from google import genai
from google.genai import types

from app.models.schemas import PromptRequest, AIResponse
from app.core.config import settings
from app.core.exceptions import AIModelError

# Initialize the Gemini API client
client = genai.Client(api_key=settings.GEMINI_API_KEY)


async def _execute_model_call(model_name: str, request: PromptRequest) -> AIResponse:
    """
    Core helper function to execute a model call using the new google.genai SDK.
    Raises AIModelError on failure.
    """
    try:
        # Prepare configuration
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=2048,
            system_instruction=request.system_role
        )

        # Execute generation
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=request.user_query,
                config=config
            )
        except Exception as e:
            # Wrap SDK errors
            raise AIModelError(f"Failed to generate content from {model_name}", {"original_error": str(e)})

        # Extract token usage
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
         raise AIModelError(f"Unexpected error in {model_name} service", {"error": str(e)})


class ChainOfThoughtAgent:
    """
    Abstracts the Chain of Thought reasoning logic.
    """
    def __init__(self, lite_model: str, flash_model: str):
        self.lite_model = lite_model
        self.flash_model = flash_model

    async def execute(self, request: PromptRequest) -> AIResponse:
        try:
            # Step 1: Analysis (Lite Model)
            analysis_request = PromptRequest(
                user_query=f"Analyze this problem step by step and break it down into logical steps. Do not solve it yet, just analyze: {request.user_query}",
                system_role="You are an analytical sub-agent. Think step by step.",
                temperature=request.temperature
            )
            sub_agent_response = await _execute_model_call(self.lite_model, analysis_request)

            # Validate analysis
            if not sub_agent_response.content or len(sub_agent_response.content) < 10:
                 raise AIModelError("Lite model failed to produce valid analysis", {"content": sub_agent_response.content})

            # Step 2: Synthesis (Flash Model)
            synthesis_query = f"Original Query: {request.user_query}\n\nAnalysis:\n{sub_agent_response.content}\n\nBased on the analysis, provide the final comprehensive answer."
            main_agent_request = PromptRequest(
                user_query=synthesis_query,
                system_role=request.system_role,
                temperature=request.temperature
            )
            final_response = await _execute_model_call(self.flash_model, main_agent_request)

            return AIResponse(
                content=final_response.content,  # Return clean final answer as main content
                tokens_used=sub_agent_response.tokens_used + final_response.tokens_used,
                model_name="chain-of-thought (lite+flash)",
                steps=[
                    {"title": "Analysis (Lite)", "content": sub_agent_response.content, "style": "cyan"},
                    {"title": "Final Answer (Flash)", "content": final_response.content, "style": "green"}
                ]
            )
        except AIModelError:
             raise
        except Exception as e:
             raise AIModelError("Chain of Thought failed", {"error": str(e)})


# Initialize global agent instance
cot_agent = ChainOfThoughtAgent(
    lite_model=settings.LITE_MODEL_NAME,
    flash_model=settings.FLASH_MODEL_NAME
)


async def call_flash_model(request: PromptRequest) -> AIResponse:
    """Call the flash model using the core helper"""
    return await _execute_model_call(settings.FLASH_MODEL_NAME, request)


async def call_lite_model(request: PromptRequest) -> AIResponse:
    """Call the lite model using the core helper"""
    return await _execute_model_call(settings.LITE_MODEL_NAME, request)


async def generate_zero_shot_response(request: PromptRequest) -> AIResponse:
    """Generate response using zero-shot approach (defaults to Flash)"""
    return await call_flash_model(request)


async def generate_response(request: PromptRequest) -> AIResponse:
    """Default entry point (defaults to Flash)"""
    return await call_flash_model(request)


async def generate_chain_of_thought_response(request: PromptRequest) -> AIResponse:
    """Wrapper for chain of thought reasoning using the agent class"""
    return await cot_agent.execute(request)
