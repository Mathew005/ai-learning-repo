import typer
import asyncio
from rich.console import Console
from rich.panel import Panel
from app.services import prompt_engine
from app.models.schemas import PromptRequest, AIResponse
from app.core.config import settings

# Initialize Typer and Rich Console
app = typer.Typer(no_args_is_help=True)
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
    with console.status(f"[{style}]{loading_text}[/{style}]"):
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
    console.print(f"Model 1: [blue]{settings.MODEL_1}[/blue]")
    console.print(f"Model 2: [cyan]{settings.MODEL_2}[/cyan]")


@app.command()
def ask(question: str, role: str = "Assistant"):
    """
    Ask a question using the default primary model (Model 1).
    """
    request_data = PromptRequest(user_query=question, system_role=role)
    asyncio.run(_run_prompt_command(
        prompt_engine.generate_response(request_data),
        title=f"Response (Model 1)",
        style="blue",
        loading_text=f"Processing: {question}..."
    ))


@app.command()
def chain_of_thought(question: str, role: str = "Assistant"):
    """
    Run chain-of-thought reasoning (Model 2 analyzes -> Model 1 synthesizes).
    """
    request_data = PromptRequest(user_query=question, system_role=role)
    asyncio.run(_run_prompt_command(
        prompt_engine.generate_chain_of_thought_response(request_data),
        title="Chain of Thought Response",
        style="purple",
        loading_text=f"Reasoning: {question}..."
    ))


@app.command()
def use_model(slot: int, question: str, role: str = "Assistant"):
    """
    Ask a question using a specific model slot (1 or 2) from settings.
    """
    request_data = PromptRequest(user_query=question, system_role=role)
    
    if slot not in [1, 2]:
        console.print("[bold red]Error:[/bold red] Slot must be 1 or 2.")
        return

    model_name = settings.MODEL_1 if slot == 1 else settings.MODEL_2
    style = "blue" if slot == 1 else "cyan"

    asyncio.run(_run_prompt_command(
        prompt_engine.call_specific_model_by_slot(slot, request_data),
        title=f"Response (Model {slot})",
        style=style,
        loading_text=f"Processing with Model {slot} ({model_name}): {question}..."
    ))


if __name__ == "__main__":
    app()