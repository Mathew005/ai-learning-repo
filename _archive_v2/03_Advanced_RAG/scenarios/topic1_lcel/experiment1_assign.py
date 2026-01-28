from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from rich.panel import Panel
from app.langchain_lab.core import LabExperiment, console
from app.langchain_lab.utils import visualize_step

class Experiment1_1_Passthrough(LabExperiment):
    name: str = "1.1 RunnablePassthrough & Assign"
    category: str = "Topic 1: LCEL"

    async def run(self):
        # 1. Concept: Real LCEL
        console.print(Panel(
            "[bold]True LCEL Composition[/bold]\n\n"
            "[yellow]chain = (Assign Joke) | (Log) | (Assign Fact) | (Log) | (Final)[/yellow]\n\n",
            title="🧪 Proper LCEL Structure",
            border_style="cyan"
        ))
        
        # 2. Setup Model
        config_google = self.get_model_config("google")
        config_ollama = self.get_model_config("ollama")
        use_ollama = "base_url" in config_ollama and config_ollama.get("model")
        
        if use_ollama:
            llm = ChatOllama(**config_ollama)
            model_name = config_ollama['model']
        else:
            llm = ChatGoogleGenerativeAI(**config_google)
            model_name = config_google['model']
            
        console.print(f"\n[dim]Using Model: {model_name}[/dim]")

        # 3. Define Components
        prompt_joke = ChatPromptTemplate.from_template("Tell me a short joke about {topic}.")
        prompt_fact = ChatPromptTemplate.from_template("Tell me a one-sentence interesting fact about {topic}.")
        prompt_summary = ChatPromptTemplate.from_template(
            "Topic: {topic}\nJoke: {joke}\nFact: {fact}\n\nCombine these into a coherent paragraph."
        )

        # 4. Build the TRUE LCEL Chain
        # Using the imported `visualize_step` for clean, readable code.
        full_chain = (
            {"topic": RunnablePassthrough()} 
            | visualize_step("0. Initial State", ["topic"])
            
            # Step 1: Assign Joke
            | RunnablePassthrough.assign(joke=prompt_joke | llm | StrOutputParser())
            | visualize_step("1. After Assign(joke)", ["joke"])
            
            # Step 2: Assign Fact
            | RunnablePassthrough.assign(fact=prompt_fact | llm | StrOutputParser())
            | visualize_step("2. After Assign(fact)", ["fact"])
            
            # Step 3: Final Synthesis
            | prompt_summary 
            | llm 
            | StrOutputParser()
        )

        # 5. Execute
        topic = "Quantum Physics"
        console.print(f"\n[bold yellow]Running Chain for: {topic}[/bold yellow]...")
        
        result = await full_chain.ainvoke(topic)
        
        console.print(Panel(result, title="🎉 Final Result", border_style="green"))
