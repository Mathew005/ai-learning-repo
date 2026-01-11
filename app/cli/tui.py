import typer
import questionary
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from app.core.config import settings
from app.services import prompt_engine
from app.services.rag_engine import RAGEngine
from app.services.embedding_provider import EmbeddingFactory
from app.services.vector_store import VectorStore
from app.models.schemas import PromptRequest

console = Console()
app = typer.Typer()

async def direct_ai_menu():
    while True:
        choice = await questionary.select(
            "🤖 Direct AI Interaction",
            choices=[
                "Ask Question (Model 1)",
                "Chain of Thought (Model 2 -> 1)",
                "Specific Model Slot",
                "Back to Main Menu"
            ]
        ).ask_async()

        if choice == "Back to Main Menu":
            break
        
        if choice == "Ask Question (Model 1)":
            q = await questionary.text("Enter your question:").ask_async()
            if q:
                with console.status("[bold green]Ask Model 1..."):
                    response = await prompt_engine.call_model_1(PromptRequest(user_query=q))
                console.print(Panel(response.content, title=f"Response ({response.model_name})", border_style="blue"))
        
        elif choice == "Chain of Thought (Model 2 -> 1)":
            q = await questionary.text("Enter your question:").ask_async()
            if q:
                with console.status("[bold purple]Reasoning..."):
                    response = await prompt_engine.generate_chain_of_thought_response(PromptRequest(user_query=q))
                for step in response.steps:
                    console.print(Panel(step['content'], title=step['title'], border_style=step.get('style', 'white')))

        elif choice == "Specific Model Slot":
            slot = await questionary.select("Select Slot", choices=["Slot 1", "Slot 2"]).ask_async()
            q = await questionary.text("Enter your question:").ask_async()
            if q:
                slot_id = 1 if slot == "Slot 1" else 2
                with console.status(f"[bold cyan]Ask Model {slot_id}..."):
                    response = await prompt_engine.call_specific_model_by_slot(slot_id, PromptRequest(user_query=q))
                console.print(Panel(response.content, title=f"Response ({response.model_name})", border_style="cyan"))
        
        console.print("") # easy spacing

# Global State
RAG_MODEL_SLOT = 1

async def rag_menu():
    global RAG_MODEL_SLOT
    while True:
        # Show current settings in menu title
        slot_name = "Model 1" if RAG_MODEL_SLOT == 1 else "Model 2"
        current_provider = settings.EMBEDDING_PROVIDER_URN.split('/')[0].capitalize()
        
        choice = await questionary.select(
            f"📚 Knowledge Base (RAG) [Active: {current_provider} | Inference: {slot_name}]",
            choices=[
                "Chat with Documents",
                "Ingest Documents",
                "View Status",
                "⚙️ Configure Settings",
                "Back to Main Menu"
            ]
        ).ask_async()

        if choice == "Back to Main Menu":
            break
            
        elif choice == "⚙️ Configure Settings":
             while True:
                # Sub-menu for RAG Configuration
                current_urn = settings.EMBEDDING_PROVIDER_URN
                slot_name = "Model 1" if RAG_MODEL_SLOT == 1 else "Model 2"
                
                sub_choice = await questionary.select(
                    f"RAG Settings [Provider: {current_urn}] [Inference: {slot_name}]",
                    choices=[
                        "Switch Embedding Provider",
                        "Change Inference Model",
                        "Back to RAG Menu"
                    ]
                ).ask_async()
                
                if sub_choice == "Back to RAG Menu":
                    break
                    
                elif sub_choice == "Change Inference Model":
                    selection = await questionary.select(
                        "Select Model for RAG Answers:",
                        choices=["Model 1 (Primary)", "Model 2 (Secondary)"]
                    ).ask_async()
                    RAG_MODEL_SLOT = 1 if "Model 1" in selection else 2
                    console.print(f"[green]RAG Inference set to use Model {RAG_MODEL_SLOT}[/green]")

                elif sub_choice == "Switch Embedding Provider":
                    # Define options from config
                    options = [
                        settings.EMBEDDING_MODEL_1,
                        settings.EMBEDDING_MODEL_2,
                        settings.EMBEDDING_MODEL_3
                    ]
                    # Filter out empty strings if any are unset
                    options = [opt for opt in options if opt]
                    
                    selection = await questionary.select("Select Provider:", choices=options).ask_async()
                    
                    if selection != current_urn:
                        settings.EMBEDDING_PROVIDER_URN = selection
                        # Reset factory cache to force reload next time a provider is requested
                        EmbeddingFactory._instance = None
                        console.print(f"[green]Switched to {selection}. Active collection will change automatically.[/green]")

        elif choice == "View Status":
            # Force reload of provider to ensure we are looking at the right collection
            EmbeddingFactory._instance = None 
            
            with console.status("[bold blue]Scanning status..."):
                 files = RAGEngine.get_source_files()
                 # Get explicit collection name for clarity
                 coll_name = VectorStore.get_collection().name
                 
            ingested_count = sum(1 for f in files if f['status'] == "INGESTED")
            total_count = len(files)
            
            table = Table(title=f"Document Status (Collection: [bold green]{coll_name}[/bold green])")
            table.add_column("Filename", style="cyan")
            table.add_column("Status", style="magenta")
            
            for f in files:
                emoji = "✅" if f['status'] == "INGESTED" else "🆕"
                table.add_row(f['name'], f"{emoji} {f['status']}")
            
            console.print(table)
            console.print(f"[dim]Tracking {ingested_count}/{total_count} files in '{coll_name}'.[/dim]")
            console.input("[dim]Press Enter to continue...[/dim]")

        elif choice == "Ingest Documents":
            EmbeddingFactory._instance = None # Safety reset
            with console.status("[bold blue]Scanning..."):
                files = RAGEngine.get_source_files()
                
            # Prepare choices for questionary
            choices = []
            for f in files:
                checked = False # Default unchecked
                title = f"{f['name']}"
                if f['status'] == "INGESTED":
                    title += " (Already Ingested)"
                choices.append(questionary.Choice(title=title, value=f['name'], checked=checked))
            
            if not choices:
                console.print("[yellow]No files found in source directory.[/yellow]")
                continue

            console.print("[bold cyan]INSTRUCTIONS: Use [Space] to select/deselect files, [Enter] to confirm.[/bold cyan]")
            selected = await questionary.checkbox("Select files to ingest:", choices=choices).ask_async()
            
            if selected:
                with console.status(f"[bold yellow]Ingesting to {current_provider} Collection..."):
                    results = RAGEngine.ingest_files(selected)
                
                console.print("\n[bold]Ingestion Results:[/bold]")
                for fname, res in results.items():
                    color = "green" if res == "Success" else "red"
                    console.print(f"[{color}]  • {fname}: {res}[/{color}]")
            else:
                console.print("\n[bold red]No files selected![/bold red] (Did you forget to press Space?)")
            
            console.input("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "Chat with Documents":
            console.print(f"[bold green]Entering RAG Chat Mode (Using {slot_name}). Type 'exit' to quit.[/bold green]")
            while True:
                q = await questionary.text("RAG Query:").ask_async()
                if q.lower() in ["exit", "quit"]:
                    break
                if not q: continue
                
                with console.status(f"[bold blue]Retrieving & Thinking ({slot_name})..."):
                    try:
                        response = await RAGEngine.generate_rag_response(q, model_slot=RAG_MODEL_SLOT)
                        console.print(Panel(response.content, title=f"RAG Answer ({response.model_name})", border_style="green"))
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")


from app.langchain_lab.registry import ExperimentRegistry

async def langchain_lab_menu():
    # Scan for new experiments every time we enter the lab
    ExperimentRegistry.scan()
    
    while True:
        categories = ExperimentRegistry.get_categories()
        
        # Add Exit option
        options = categories + ["Back to Main Menu"]
        
        category = await questionary.select(
            "🦜 LangChain Lab",
            choices=options
        ).ask_async()
        
        if category == "Back to Main Menu":
            break
            
        # Select Experiment in Category
        experiments = ExperimentRegistry.get_experiments_in_category(category)
        exp_choices = [questionary.Choice(title=e.name, value=e) for e in experiments]
        exp_choices.append(questionary.Choice(title="Back", value="back"))
        
        selected_exp_cls = await questionary.select(
            f"Select Experiment ({category})",
            choices=exp_choices
        ).ask_async()
        
        if selected_exp_cls == "back":
            continue
            
        # Run Experiment
        exp_instance = selected_exp_cls()
        console.print(Panel(f"Running: [bold]{exp_instance.name}[/bold]", style="bold magenta"))
        
        try:
            await exp_instance.run()
        except Exception as e:
            console.print(f"[bold red]Experiment Failed:[/bold red] {e}")
            
        console.input("\n[dim]Press Enter to continue...[/dim]")

async def main_loop():
    console.print(Panel(f"[bold blue]Welcome to {settings.PROJECT_NAME} TUI[/bold blue]", expand=False))
    
    while True:
        try:
            choice = await questionary.select(
                "Main Menu",
                choices=[
                    "🤖 Direct AI Interaction",
                    "📚 Knowledge Base (RAG)",
                    "🦜 LangChain Lab",
                    "❌ Exit"
                ]
            ).ask_async()
            
            if choice == "❌ Exit":
                console.print("Goodbye!")
                break
                
            if choice == "🤖 Direct AI Interaction":
                await direct_ai_menu()
            elif choice == "📚 Knowledge Base (RAG)":
                await rag_menu()
            elif choice == "🦜 LangChain Lab":
                await langchain_lab_menu()
                
        except Exception as e:
            console.print(f"[red]Application Error: {e}[/red]")
            break

@app.command()
def start():
    asyncio.run(main_loop())

if __name__ == "__main__":
    app()
