"""
CocoIndex Flow Definition - REFERENCE IMPLEMENTATION

Original source: project/coco_app/flow.py
Archived for future implementation of "Run CocoIndex" feature.

This file shows how to define a CocoIndex flow for live document processing.
"""

from cocoindex import FlowBuilder, DataScope, op
import os
from cocoindex.sources import LocalFile
from datetime import timedelta
import requests
from .pdf_parser import parse_pdf_pages
from .chroma_sink import write_to_chroma

def get_ollama_embedding(text, model="nomic-embed-text"):
    try:
        url = "http://localhost:11434/api/embeddings"
        resp = requests.post(url, json={"model": model, "prompt": text})
        if resp.status_code == 200:
            return resp.json()["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
    return []

# Define the processing function with @op.function decorator
@op.function()
def process_pdf_by_filename(filename: str) -> int:
    """Process PDF: parse, embed, and save to Chroma"""
    # Reconstruct full path from filename
    file_path = os.path.join(os.getcwd(), "data", "pdfs", filename)
    print(f"📄 Processing {filename}...")
    
    try:
        chunks = parse_pdf_pages(file_path)
        batch = []
        
        for chunk in chunks:
            vec = get_ollama_embedding(chunk.text)
            if vec:
                batch.append({
                    "content": chunk.text,
                    "embedding": vec,
                    "metadata": chunk.metadata
                })
        
        if batch:
            ids = write_to_chroma(batch)
            print(f"💾 Persisted {len(ids)} chunks from {filename}")
            return len(ids)
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
    
    return 0

def ingestion_flow(flow_builder: FlowBuilder, data_scope: DataScope):
    # Add source - LocalFile outputs rows with 'filename' and 'content' fields
    data_scope["files"] = flow_builder.add_source(
        LocalFile(path=os.path.join(os.getcwd(), "data", "pdfs")),
        refresh_interval=timedelta(seconds=5)
    )
    
    # Use the row() context manager for field access
    with data_scope["files"].row() as file:
        # Apply transform to the filename field
        # The function receives the string value, not DataSlice
        file["chunk_count"] = file["filename"].transform(process_pdf_by_filename)
