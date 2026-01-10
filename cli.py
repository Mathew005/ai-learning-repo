import typer
import asyncio
from rich.console import Console
from rich.panel import Panel
from app.services import prompt_engine
from app.models.schemas import PromptRequest

from app.core.config import settings

# Initialize Typer and Rich Console
app = typer.Typer()
console = Console()

@app.command()
def hello():
    """
    Test if the CLI is working.
    """
    console.print(f"[bold green]{settings.PROJECT_NAME} CLI is online.[/bold green]")

@app.command()
def ask(question: str, role: str = "Assistant"):
    """
    Send a direct question to the AI logic (Bypassing the API).
    """
    # 1. Create the data object (The same one the API uses!)
    request_data = PromptRequest(
        user_query=question,
        system_role=role
    )

    console.print(f"[bold blue]Processing:[/bold blue] {question}...")

    # 2. Call the service logic
    # Since services are async, we use asyncio.run to execute them in the terminal
    try:
        response = asyncio.run(prompt_engine.generate_response(request_data))
        
        # 3. Print the result nicely
        console.print(Panel(
            response.content, 
            title=f"AI Response ({response.model_name})", 
            border_style="green"
        ))
        console.print(f"[italic dim]Tokens used: {response.tokens_used}[/italic dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    app()