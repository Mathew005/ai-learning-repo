"""
PDF Parser - REFERENCE IMPLEMENTATION

Original source: project/coco_app/pdf_parser.py
Archived for future implementation of "Run CocoIndex" feature.

This file shows how to parse PDFs and create semantic chunks with metadata.
"""

from PyPDF2 import PdfReader
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class PdfChunk:
    text: str
    metadata: Dict[str, Any]

# Chunking settings
CHUNK_SIZE = 500  # Target chunk size in characters
CHUNK_OVERLAP = 50  # Overlap between chunks to preserve context

def parse_pdf_pages(file_path: str) -> List[PdfChunk]:
    """
    Reads a PDF and returns semantic chunks with natural boundaries.
    Uses RecursiveCharacterTextSplitter for clean splits.
    """
    chunks = []
    
    # Text splitter that respects natural boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
        length_function=len,
    )
    
    try:
        reader = PdfReader(file_path)
        file_name = file_path.split("/")[-1]
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                # Split page content into semantic chunks
                page_chunks = splitter.split_text(text.strip())
                
                for chunk_num, chunk_text in enumerate(page_chunks, 1):
                    if len(chunk_text.strip()) > 20:  # Skip tiny fragments
                        chunks.append(PdfChunk(
                            text=chunk_text.strip(),
                            metadata={
                                "source_type": "pdf",
                                "original_name": file_name,
                                "page_number": page_num,
                                "chunk_number": chunk_num
                            }
                        ))
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
    
    return chunks
