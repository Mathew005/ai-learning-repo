# CoCo RAG

A local RAG (Retrieval-Augmented Generation) system that answers questions from your PDF documents with citations.

## What It Does

1. **Ingests PDFs** - Watches a folder for PDF files, extracts text, splits into chunks
2. **Creates embeddings** - Converts text chunks to vectors using Ollama
3. **Stores in ChromaDB** - Persists embeddings locally for fast retrieval
4. **Answers questions** - Uses retrieved context to generate answers with source citations

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally with models:
  - `nomic-embed-text` (for embeddings)
  - `gemma:2b` (for local generation, optional)
- Google API key (if using Google Gemini)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull nomic-embed-text
ollama pull gemma:2b
```

## Configuration

Edit `.env` to choose your LLM:

```env
# Use local Ollama
LLM_PROVIDER=ollama

# Or use Google Gemini
LLM_PROVIDER=google
GOOGLE_API_KEY=your-api-key
```

## Usage

### 1. Start the ingestion service

```bash
python main_service.py
```

This watches `data/pdfs/` for new files and processes them automatically.

### 2. Add PDF files

Drop PDF files into `data/pdfs/`. They will be processed within 5 seconds.

### 3. Ask questions

```bash
# Single question
python -c "from rag_app.chat import chat; print(chat('What is machine learning?'))"

# Interactive mode
python rag_app/chat.py
```

## Project Structure

```
project/
├── main_service.py      # Starts ingestion pipeline
├── config.py            # LLM model configuration
├── .env                 # API keys and settings
├── coco_app/
│   ├── flow.py          # CoCo Index ingestion flow
│   ├── pdf_parser.py    # PDF text extraction and chunking
│   └── chroma_sink.py   # ChromaDB storage
├── rag_app/
│   └── chat.py          # Question answering with citations
└── data/
    └── pdfs/            # Drop PDF files here
```

## Output Example

```
Q: What is the F1 score?

A: The F1 Score is the harmonic mean of precision and recall [1].

---
**Sources:**
[1] machine_learning_guide.pdf, Page 3
    "Model Evaluation Metrics Evaluating ML models requires..."
```

## Switching Models

In `config.py`, available models are:

```python
class Models:
    OLLAMA_GEMMA = "ollama/gemma:2b"      # Local, fast
    GOOGLE_GEMINI = "google/gemini-2.0-flash"  # API, smarter
```

Set `LLM_PROVIDER` in `.env` to switch between them.
