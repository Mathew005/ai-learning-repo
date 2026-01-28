# AI-Mastery

A Python monorepo for learning AI roles and building AI systems. The project is divided into independent phases, each contained in its own directory with separate dependencies.

## Structure

### [01_Fundamentals](./01_Fundamentals)
*   **Focus**: LLM API Basics (OpenAI, Gemini, Ollama), Prompt Engineering.
*   **Key Scripts**: `01_gemini_basic.py`, `02_ollama_basic.py`.

### [02_Basic_RAG](./02_Basic_RAG)
*   **Focus**: Basic RAG Pipeline (Ingest -> Embed -> Retrive -> Generate).
*   **Tech**: FastAPI, ChromaDB, Typer CLI.
*   **Status**: Fully functional RAG application.

### [03_Advanced_RAG](./03_Advanced_RAG)
*   **Focus**: Advanced Techniques (LCEL, Graph, Hybrid Search).
*   **Tech**: LangChain, LangGraph.

### [04_Agents](./04_Agents)
*   **Focus**: Single Agent patterns (ReAct, Tools).
*   **Status**: Planned.

### [05_Multi_Agent_Systems](./05_Multi_Agent_Systems)
*   **Focus**: Orchestration (CrewAI, AutoGen).
*   **Status**: Planned.

### [06_Optimization](./06_Optimization)
*   **Focus**: Evals and Monitoring.
*   **Status**: Planned.

## Setup

Each folder is a standalone project. To work on a specific phase:

1.  Navigate to the folder: `cd 01_Fundamentals`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Follow the inner `README.md` instructions.
