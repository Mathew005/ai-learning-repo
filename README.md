# AI-Mastery

A Python monorepo for learning AI roles and building AI systems. It connects specific learning modules (Prompt Engineering, RAG, Agents) to a shared backend using FastAPI and a CLI using Typer.

## Roadmap & Progress

- [x] **Phase 1: Fundamentals (LLM & APIs)**
    - [x] LLM Fundamentals
    - [x] Prompt Engineering (Zero-shot to Chain-of-Thought)
    - [x] LLM APIs Integration (OpenAI, Claude, Gemini, Ollama)

- [x] **Phase 2: RAG Systems (Basic)**
    - [x] Vector Databases (ChromaDB)
    - [x] Embeddings & Semantic Search
    - [x] In-Memory Conversation History
    - [x] Basic Document Ingestion (TXT)

- [ ] **Phase 3: Advanced RAG & LangChain Mastery**
    - [x] Multi-Source RAG Q&A System with Citations (PDF Support)
    - [ ] LangChain Mastery (LCEL, Runtime Config)
    - [ ] Advanced RAG: Query Transformation (Multi-Query, History Rewriting)
    - [ ] Advanced RAG: Re-ranking (Cross-Encoders)
    - [ ] Memory Management (Persistent Sessions)

- [ ] **Phase 4: AI Agents**
    - [ ] ReAct Pattern (Reasoning + Acting)
    - [ ] LangGraph Implementation
    - [ ] Tool Calling & Function Execution

- [ ] **Phase 5: Multi-Agent Systems & Production**
    - [ ] Multi-agent Orchestration (CrewAI / AutoGen / Hierarchical Teams)
    - [ ] Multi-modal AI (Vision, Audio)
    - [ ] Production Deployment & Integration

- [ ] **Phase 6: Optimization & Evaluation**
    - [ ] Evaluation Frameworks (RAGAS, TruLens)
    - [ ] Security (Prompt Injection Defense)
    - [ ] Cost Optimization Strategies

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
# AI Integration Specialist Repo

A professional AI backend implementing a modular **Retrieval-Augmented Generation (RAG)** pipeline with a **Unified TUI** and **REST API**.

## 🚀 Key Features

*   **Multi-Model Architecture**: Seamlessly switch between **Gemini**, **Ollama**, and **HuggingFace** models.
*   **Basic RAG Pipeline**: Ingest documentation and query it using your preferred LLM.
*   **Unified TUI**: Interactive terminal menu for all operations (Chat, Ingest, Config).
*   **REST API**: Full programmatic access via FastAPI (`/rag/query`, `/rag/ingest`).
*   **Provider Isolation**: Separate vector collections for each embedding provider to prevent conflicts.

## 🛠️ Quick Start

### 1. Setup
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file:
```ini
GEMINI_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
# See config.py for full list of options
```

### 3. Run the TUI (Interactive Mode)
The recommended way to use the application:
```bash
python cli.py start
```
*   **Knowledge Base**: Access `Ingest`, `Chat`, and `Configure Settings` (switch providers/models) all from the RAG menu.

### 4. Run the API (Server Mode)
```bash
python main.py
```
*   Docs: `http://localhost:8000/docs`
*   RAG API: `http://localhost:8000/rag`

## 📚 Knowledge Base (RAG)
Place your `.txt` documents in `data/source_documents`.
*   **Ingest**: Use the TUI or API to index them.
*   **Embeddings**: Supports Google (`gemini-embedding-001`), Ollama (`embeddinggemma`), and HuggingFace (`all-MiniLM-L6-v2`).

## 🏗️ Architecture
*   **Core**: FastAPI, Pydantic
*   **Vector Store**: ChromaDB (Persistent)
*   **Orchestration**: Custom `RAGEngine` & `PromptEngine`
*   **Interface**: `Typer` & `Questionary` (TUI)

**Note**: This implements a *Basic RAG* architecture (simple chunking, direct retrieval). Advanced techniques (re-ranking, hybrid search) are not yet implemented.
