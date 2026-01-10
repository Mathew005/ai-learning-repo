import typer
import asyncio
from rich.console import Console
from rich.panel import Panel
from app.services import prompt_engine
from app.models.schemas import PromptRequest, AIResponse

from app.core.config import settings

# Initialize Typer and Rich Console
app = typer.Typer()
console = Console()


async def _run_prompt_command(
    coroutine,
    title: str,
    style: str,
    loading_text: str
):
    """
    Helper to run a prompt command with standard UI handling.
    """
    console.print(f"[{style}]{loading_text}[/{style}]")

    try:
        response: AIResponse = await coroutine

        if response.steps:
            # Render multi-step response
            for step in response.steps:
                step_title = step.get("title", "Step")
                step_content = step.get("content", "")
                step_style = step.get("style", style)
                
                console.print(Panel(
                    step_content,
                    title=f"{step_title} ({response.model_name})",
                    border_style=step_style
                ))
        else:
            # Render single-step response
            console.print(Panel(
                response.content,
                title=f"{title} ({response.model_name})",
                border_style=style
            ))
            
        console.print(f"[italic dim]Tokens used: {response.tokens_used}[/italic dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@app.command()
def hello():
    """
    Test if the CLI is working.
    """
    console.print(f"[bold green]{settings.PROJECT_NAME} CLI is online.[/bold green]")


@app.command()
def ask(question: str, role: str = "Assistant"):
    """
    Send a direct question to the AI logic (Bypassing the API) - defaults to flash model.
    """
    request_data = PromptRequest(user_query=question, system_role=role)
    asyncio.run(_run_prompt_command(
        prompt_engine.generate_response(request_data),
        title="AI Response",
        style="green",
        loading_text=f"Processing: {question}..."
    ))


@app.command()
def flash_ask(question: str, role: str = "Assistant"):
    """
    Send a direct question to the AI using the flash model.
    """
    request_data = PromptRequest(user_query=question, system_role=role)
    asyncio.run(_run_prompt_command(
        prompt_engine.call_flash_model(request_data),
        title="Flash Model Response",
        style="blue",
        loading_text=f"Processing with flash model: {question}..."
    ))


@app.command()
def lite_ask(question: str, role: str = "Assistant"):
    """
    Send a direct question to the AI using the lite model.
    """
    request_data = PromptRequest(user_query=question, system_role=role)
    asyncio.run(_run_prompt_command(
        prompt_engine.call_lite_model(request_data),
        title="Lite Model Response",
        style="cyan",
        loading_text=f"Processing with lite model: {question}..."
    ))


@app.command()
def chain_of_thought(question: str, role: str = "Assistant"):
    """
    Process a question using chain of thought reasoning with sub-agent approach.
    """
    request_data = PromptRequest(user_query=question, system_role=role)
    asyncio.run(_run_prompt_command(
        prompt_engine.generate_chain_of_thought_response(request_data),
        title="Chain of Thought Response",
        style="purple",
        loading_text=f"Processing with chain of thought: {question}..."
    ))


@app.command()
def zero_shot(question: str, role: str = "Assistant"):
    """
    Process a question using zero-shot approach with flash model.
    """
    request_data = PromptRequest(user_query=question, system_role=role)
    asyncio.run(_run_prompt_command(
        prompt_engine.generate_zero_shot_response(request_data),
        title="Zero-Shot Response",
        style="yellow",
        loading_text=f"Processing with zero-shot approach: {question}..."
    ))


if __name__ == "__main__":
    app()