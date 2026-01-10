# AI-Mastery

A Python monorepo for learning AI roles and building AI systems. It connects specific learning modules (Prompt Engineering, RAG, Agents) to a shared backend using FastAPI and a CLI using Typer.

## Roadmap & Progress

- [x] **1. Fundamentals**
    - [x] LLM Fundamentals
    - [x] Prompt Engineering (Zero-shot to Chain-of-Thought)
    - [x] LLM APIs Integration (OpenAI, Claude, Gemini)

- [ ] **2. RAG Systems**
    - [ ] Vector Databases (Pinecone, Weaviate, Chroma)
    - [ ] Embeddings & Semantic Search
    - [ ] LangChain Mastery & Memory Management
    - [ ] Advanced RAG (Query Transformation, Reranking)

- [ ] **3. Milestone Project**
    - [ ] Multi-Source RAG Q&A System with Citations

- [ ] **4. AI Agents**
    - [ ] ReAct Pattern
    - [ ] LangGraph
    - [ ] Tool Calling & Function Execution

- [ ] **5. Advanced & Production**
    - [ ] Multi-agent Systems (CrewAI, AutoGen)
    - [ ] Multi-modal AI
    - [ ] Evaluation (RAGAS, TruLens)
    - [ ] Security & Cost Optimization

## Tech Stack

*   **Python 3.11+**
*   **FastAPI** (API)
*   **Typer** (CLI)
*   **LangChain** (Logic)
*   **Pydantic** (Validation)

## Setup (Windows)

1.  **Install**
    ```powershell
    pip install -r requirements.txt
    ```

2.  **Config**
    Create a `.env` file in the root:
    ```ini
    GEMINI_API_KEY="..."
    ```

## Usage

### 1. HTTP API
Runs the server and Swagger UI.

```powershell
python main.py
```
*   Docs: `http://localhost:8000/docs`

### 2. CLI
Runs specific modules directly from the terminal.

```powershell
# Check status
python cli.py hello

# Send query
python cli.py ask "Explain recursion"
```

## Structure

*   `app/services/`: Pure Python logic (AI Chains).
*   `app/routers/`: FastAPI endpoints.
*   `main.py`: Web server entry point.
*   `cli.py`: Terminal entry point.
