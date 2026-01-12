import typer
import questionary
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from app.core.config import settings
from app.services import prompt_engine
from app.services.rag_engine import RAGEngine
from app.services.qa_engine import QAEngine
from app.services.cocoindex_service import CocoIndexService
from app.services.embedding_provider import EmbeddingFactory
from app.services.vector_store import VectorStore
from app.services.model_registry import ModelRegistry
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
QA_MODEL_SLOT = 1

async def basic_rag_menu():
    """Submenu for Basic RAG (text documents)."""
    global RAG_MODEL_SLOT
    while True:
        slot_name = "Model 1" if RAG_MODEL_SLOT == 1 else "Model 2"
        current_provider = settings.EMBEDDING_PROVIDER_URN.split('/')[0].capitalize()
        
        choice = await questionary.select(
            f"📄 Basic RAG [Provider: {current_provider} | Model: {slot_name}]",
            choices=[
                "Chat with Documents",
                "Ingest Documents",
                "View Status",
                "Back"
            ]
        ).ask_async()
        
        if choice == "Back":
            break
        
        elif choice == "View Status":
            EmbeddingFactory._instance = None
            with console.status("[bold blue]Scanning status..."):
                files = RAGEngine.get_source_files()
                coll_name = VectorStore.get_collection().name
            
            ingested_count = sum(1 for f in files if f['status'] == "INGESTED")
            total_count = len(files)
            
            table = Table(title=f"Basic RAG Status (Collection: [bold green]{coll_name}[/bold green])")
            table.add_column("Filename", style="cyan")
            table.add_column("Status", style="magenta")
            
            for f in files:
                emoji = "✅" if f['status'] == "INGESTED" else "🆕"
                table.add_row(f['name'], f"{emoji} {f['status']}")
            
            console.print(table)
            console.print(f"[dim]Tracking {ingested_count}/{total_count} files in '{coll_name}'.[/dim]")
            console.print(f"[dim]Source directory: data/source_documents/[/dim]")
            console.input("[dim]Press Enter to continue...[/dim]")
        
        elif choice == "Ingest Documents":
            EmbeddingFactory._instance = None
            with console.status("[bold blue]Scanning..."):
                files = RAGEngine.get_source_files()
            
            choices = []
            for f in files:
                title = f"{f['name']}"
                if f['status'] == "INGESTED":
                    title += " (Already Ingested)"
                choices.append(questionary.Choice(title=title, value=f['name'], checked=False))
            
            if not choices:
                console.print("[yellow]No files found in data/source_documents/[/yellow]")
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
            console.print(f"[bold green]Entering Basic RAG Chat (Using {slot_name}). Type 'exit' to quit.[/bold green]")
            while True:
                q = await questionary.text("Query:").ask_async()
                if q.lower() in ["exit", "quit"]:
                    break
                if not q:
                    continue
                
                with console.status(f"[bold blue]Retrieving & Thinking ({slot_name})..."):
                    try:
                        response = await RAGEngine.generate_rag_response(q, model_slot=RAG_MODEL_SLOT)
                        console.print(Panel(response.content, title=f"RAG Answer ({response.model_name})", border_style="green"))
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")


async def multisource_qa_menu():
    """Submenu for Multi-Source Q&A (PDF with citations)."""
    global QA_MODEL_SLOT
    while True:
        slot_name = "Model 1" if QA_MODEL_SLOT == 1 else "Model 2"
        current_provider = settings.EMBEDDING_PROVIDER_URN.split('/')[0].capitalize()
        
        choice = await questionary.select(
            f"📑 Multi-Source Q&A [Provider: {current_provider} | Model: {slot_name}]",
            choices=[
                "Ask Question (with Citations)",
                "Ingest PDFs",
                "View Status",
                "Back"
            ]
        ).ask_async()
        
        if choice == "Back":
            break
        
        elif choice == "View Status":
            EmbeddingFactory._instance = None
            with console.status("[bold blue]Scanning status..."):
                files = QAEngine.get_source_files()
                coll_name = QAEngine.get_collection().name
            
            ingested_count = sum(1 for f in files if f['status'] == "INGESTED")
            total_count = len(files)
            
            table = Table(title=f"Multi-Source Q&A Status (Collection: [bold green]{coll_name}[/bold green])")
            table.add_column("Filename", style="cyan")
            table.add_column("Status", style="magenta")
            
            for f in files:
                emoji = "✅" if f['status'] == "INGESTED" else "🆕"
                table.add_row(f['name'], f"{emoji} {f['status']}")
            
            console.print(table)
            console.print(f"[dim]Tracking {ingested_count}/{total_count} PDF files in '{coll_name}'.[/dim]")
            console.print(f"[dim]Source directory: data/qa_documents/[/dim]")
            console.input("[dim]Press Enter to continue...[/dim]")
        
        elif choice == "Ingest PDFs":
            EmbeddingFactory._instance = None
            with console.status("[bold blue]Scanning for PDFs..."):
                files = QAEngine.get_source_files()
            
            choices = []
            for f in files:
                title = f"{f['name']}"
                if f['status'] == "INGESTED":
                    title += " (Already Ingested)"
                choices.append(questionary.Choice(title=title, value=f['name'], checked=False))
            
            if not choices:
                console.print("[yellow]No PDF files found in data/qa_documents/[/yellow]")
                console.print("[dim]Place your PDF files in the data/qa_documents/ folder.[/dim]")
                continue
            
            console.print("[bold cyan]INSTRUCTIONS: Use [Space] to select/deselect PDFs, [Enter] to confirm.[/bold cyan]")
            selected = await questionary.checkbox("Select PDFs to ingest:", choices=choices).ask_async()
            
            if selected:
                with console.status(f"[bold yellow]Ingesting PDFs to {current_provider} Collection..."):
                    results = QAEngine.ingest_files(selected)
                
                console.print("\n[bold]Ingestion Results:[/bold]")
                for fname, res in results.items():
                    color = "green" if "Success" in res else "red"
                    console.print(f"[{color}]  • {fname}: {res}[/{color}]")
            else:
                console.print("\n[bold red]No PDFs selected![/bold red]")
            
            console.input("\n[dim]Press Enter to continue...[/dim]")
        
        elif choice == "Ask Question (with Citations)":
            console.print(f"[bold green]Multi-Source Q&A (Using {slot_name}). Type 'exit' to quit.[/bold green]")
            console.print("[dim]Answers will include citations with source file, page number, and excerpt.[/dim]\n")
            
            while True:
                q = await questionary.text("Your Question:").ask_async()
                if q.lower() in ["exit", "quit"]:
                    break
                if not q:
                    continue
                
                with console.status(f"[bold blue]Searching documents & generating answer..."):
                    try:
                        qa_response = await QAEngine.ask(q, model_slot=QA_MODEL_SLOT)
                        formatted = QAEngine.format_response_with_sources(qa_response)
                        console.print(Panel(formatted, title=f"Answer ({qa_response.model_name})", border_style="green"))
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")


async def rag_menu():
    """Main RAG menu with submenus."""
    global RAG_MODEL_SLOT, QA_MODEL_SLOT
    
    while True:
        slot_name = "Model 1" if RAG_MODEL_SLOT == 1 else "Model 2"
        current_provider = settings.EMBEDDING_PROVIDER_URN.split('/')[0].capitalize()
        
        choice = await questionary.select(
            f"📚 Knowledge Base (RAG) [Active: {current_provider} | Inference: {slot_name}]",
            choices=[
                "📄 Basic RAG",
                "📑 Multi-Source Q&A",
                "🚀 Run CocoIndex",
                "⚙️ Configure Settings",
                "Back to Main Menu"
            ]
        ).ask_async()

        if choice == "Back to Main Menu":
            break
        
        elif choice == "📄 Basic RAG":
            await basic_rag_menu()
        
        elif choice == "📑 Multi-Source Q&A":
            await multisource_qa_menu()
        
        elif choice == "🚀 Run CocoIndex":
            status = CocoIndexService.get_status()
            if status["available"]:
                console.print("[yellow]CocoIndex is available but not yet configured.[/yellow]")
            else:
                console.print(Panel(
                    "[bold yellow]🚧 Coming Soon[/bold yellow]\n\n"
                    "CocoIndex provides live document watching and automatic ingestion.\n\n"
                    "To enable:\n"
                    "1. Install: [cyan]pip install cocoindex pgserver[/cyan]\n"
                    "2. Implementation reference: [dim]app/services/cocoindex_reference/[/dim]",
                    title="🚀 CocoIndex Integration",
                    border_style="yellow"
                ))
            console.input("[dim]Press Enter to continue...[/dim]")
            
        elif choice == "⚙️ Configure Settings":
            while True:
                current_urn = settings.EMBEDDING_PROVIDER_URN
                rag_slot = "Model 1" if RAG_MODEL_SLOT == 1 else "Model 2"
                qa_slot = "Model 1" if QA_MODEL_SLOT == 1 else "Model 2"
                
                sub_choice = await questionary.select(
                    f"RAG Settings [Provider: {current_urn}]",
                    choices=[
                        "Switch Embedding Provider",
                        f"Change Basic RAG Model (Current: {rag_slot})",
                        f"Change Q&A Model (Current: {qa_slot})",
                        "Back to RAG Menu"
                    ]
                ).ask_async()
                
                if sub_choice == "Back to RAG Menu":
                    break
                    
                elif "Change Basic RAG Model" in sub_choice:
                    selection = await questionary.select(
                        "Select Model for Basic RAG:",
                        choices=["Model 1 (Primary)", "Model 2 (Secondary)"]
                    ).ask_async()
                    RAG_MODEL_SLOT = 1 if "Model 1" in selection else 2
                    console.print(f"[green]Basic RAG set to use Model {RAG_MODEL_SLOT}[/green]")
                
                elif "Change Q&A Model" in sub_choice:
                    selection = await questionary.select(
                        "Select Model for Multi-Source Q&A:",
                        choices=["Model 1 (Primary)", "Model 2 (Secondary)"]
                    ).ask_async()
                    QA_MODEL_SLOT = 1 if "Model 1" in selection else 2
                    console.print(f"[green]Multi-Source Q&A set to use Model {QA_MODEL_SLOT}[/green]")

                elif sub_choice == "Switch Embedding Provider":
                    options = [
                        settings.EMBEDDING_MODEL_1,
                        settings.EMBEDDING_MODEL_2,
                        settings.EMBEDDING_MODEL_3
                    ]
                    options = [opt for opt in options if opt]
                    
                    selection = await questionary.select("Select Provider:", choices=options).ask_async()
                    
                    if selection != current_urn:
                        settings.EMBEDDING_PROVIDER_URN = selection
                        EmbeddingFactory._instance = None
                        console.print(f"[green]Switched to {selection}. Active collection will change automatically.[/green]")


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

async def model_settings_menu():
    """Model Settings menu for dynamic model selection."""
    registry = ModelRegistry.instance()
    
    while True:
        llm1 = registry.get_active_llm(1) or "Not Set"
        llm2 = registry.get_active_llm(2) or "Not Set"
        embed = registry.get_active_embedding() or "Not Set"
        
        choice = await questionary.select(
            "⚙️ Model Settings",
            choices=[
                "🔄 Refresh Available Models",
                f"Set LLM Slot 1 ({llm1})",
                f"Set LLM Slot 2 ({llm2})",
                f"Set Embedding ({embed})",
                "View All Available Models",
                "Back to Main Menu"
            ]
        ).ask_async()
        
        if choice == "Back to Main Menu":
            break
        
        elif choice == "🔄 Refresh Available Models":
            with console.status("[bold blue]Discovering models..."):
                registry.refresh_models(
                    gemini_api_key=settings.GEMINI_API_KEY,
                    ollama_base_url=settings.OLLAMA_BASE_URL
                )
            llms = registry.get_available_llms()
            embeds = registry.get_available_embeddings()
            console.print(f"[green]✅ Found {len(llms)} LLMs and {len(embeds)} embedding models.[/green]")
            console.input("[dim]Press Enter to continue...[/dim]")
        
        elif choice == "View All Available Models":
            llms = registry.get_available_llms()
            embeds = registry.get_available_embeddings()
            
            table = Table(title="Available Models")
            table.add_column("Type", style="cyan")
            table.add_column("URN", style="white")
            table.add_column("Provider", style="magenta")
            table.add_column("Size", style="dim")
            
            for m in llms:
                size = f"{m.size_gb:.1f}GB" if m.size_gb else "-"
                table.add_row("LLM", m.urn, m.provider, size)
            for m in embeds:
                size = f"{m.size_gb:.1f}GB" if m.size_gb else "-"
                table.add_row("Embedding", m.urn, m.provider, size)
            
            console.print(table)
            console.input("[dim]Press Enter to continue...[/dim]")
        
        elif "Set LLM Slot 1" in choice:
            llms = registry.get_available_llms()
            if not llms:
                console.print("[yellow]No LLM models available. Run Refresh first.[/yellow]")
                continue
            choices = [questionary.Choice(title=m.urn, value=m.urn) for m in llms]
            selection = await questionary.select("Select LLM for Slot 1:", choices=choices).ask_async()
            if selection:
                registry.set_active_llm(1, selection)
                console.print(f"[green]✅ Slot 1 set to {selection}[/green]")
        
        elif "Set LLM Slot 2" in choice:
            llms = registry.get_available_llms()
            if not llms:
                console.print("[yellow]No LLM models available. Run Refresh first.[/yellow]")
                continue
            choices = [questionary.Choice(title=m.urn, value=m.urn) for m in llms]
            selection = await questionary.select("Select LLM for Slot 2:", choices=choices).ask_async()
            if selection:
                registry.set_active_llm(2, selection)
                console.print(f"[green]✅ Slot 2 set to {selection}[/green]")
        
        elif "Set Embedding" in choice:
            embeds = registry.get_available_embeddings()
            if not embeds:
                console.print("[yellow]No embedding models available. Run Refresh first.[/yellow]")
                continue
            choices = [questionary.Choice(title=m.urn, value=m.urn) for m in embeds]
            selection = await questionary.select("Select Embedding Model:", choices=choices).ask_async()
            if selection:
                registry.set_active_embedding(selection)
                EmbeddingFactory.reset()  # Reset cache to use new embedding
                console.print(f"[green]✅ Embedding set to {selection}[/green]")


async def main_loop():
    # Initialize Model Registry at startup
    console.print("[dim]Initializing model registry...[/dim]")
    registry = ModelRegistry.instance()
    registry.initialize(
        gemini_api_key=settings.GEMINI_API_KEY,
        ollama_base_url=settings.OLLAMA_BASE_URL
    )
    
    llms = registry.get_available_llms()
    embeds = registry.get_available_embeddings()
    console.print(f"[dim]Found {len(llms)} LLMs and {len(embeds)} embedding models.[/dim]\n")
    
    console.print(Panel(f"[bold blue]Welcome to {settings.PROJECT_NAME} TUI[/bold blue]", expand=False))
    
    while True:
        try:
            choice = await questionary.select(
                "Main Menu",
                choices=[
                    "🤖 Direct AI Interaction",
                    "📚 Knowledge Base (RAG)",
                    "🦜 LangChain Lab",
                    "⚙️ Model Settings",
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
            elif choice == "⚙️ Model Settings":
                await model_settings_menu()
                
        except Exception as e:
            console.print(f"[red]Application Error: {e}[/red]")
            break

@app.command()
def start():
    asyncio.run(main_loop())

if __name__ == "__main__":
    app()
