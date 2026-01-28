# 01_Fundamentals

This project focuses on the basics of connecting to Large Language Models (LLMs) via their APIs.

## Scripts
*   `01_gemini_basic.py`: Connects to Google's Gemini API (requires `GEMINI_API_KEY` in `.env`).
*   `02_ollama_basic.py`: Connects to a local Ollama instance (default `http://localhost:11434`).

## Setup
1.  Install requirements: `pip install -r requirements.txt`
2.  Create a `.env` file with your `GEMINI_API_KEY`.
3.  Ensure Ollama is running if using the Ollama script.
