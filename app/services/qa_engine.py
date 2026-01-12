"""
Multi-Source Q&A Engine with PDF support and citations.

This engine provides document Q&A with:
- PDF parsing and semantic chunking
- Page-level metadata tracking
- Citation-aware responses with source attribution
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from PyPDF2 import PdfReader

from app.core.config import settings
from app.services.embedding_provider import EmbeddingFactory
from app.services import prompt_engine
from app.models.schemas import PromptRequest
from app.utils.text_splitter import RecursiveCharacterTextSplitter


# --- Data Classes ---

@dataclass
class Citation:
    """A citation reference with source details."""
    index: int
    source: str
    page: int
    excerpt: str


@dataclass
class QAResponse:
    """Response from the Q&A engine with citations."""
    answer: str
    citations: List[Citation]
    model_name: str


@dataclass
class PdfChunk:
    """A chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any]


# --- Constants ---

QA_COLLECTION_NAME = "qa_collection"
QA_DOCUMENTS_DIR = "./data/qa_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# --- Q&A Engine ---

class QAEngine:
    """
    Multi-Source Q&A Engine.
    
    Manages a separate ChromaDB collection for PDF documents with
    rich metadata for citations.
    """
    
    _client = None
    
    @classmethod
    def get_client(cls):
        """Get or create ChromaDB client."""
        if not cls._client:
            cls._client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        return cls._client
    
    @classmethod
    def get_collection(cls):
        """Get the Q&A collection (separate from Basic RAG)."""
        client = cls.get_client()
        provider = EmbeddingFactory.get_provider()
        # Include provider name to avoid embedding dimension conflicts
        collection_name = f"{QA_COLLECTION_NAME}_{provider.provider_name}"
        return client.get_or_create_collection(name=collection_name)
    
    @classmethod
    def get_source_dir(cls) -> str:
        """Get the source documents directory, creating if needed."""
        if not os.path.exists(QA_DOCUMENTS_DIR):
            os.makedirs(QA_DOCUMENTS_DIR)
        return QA_DOCUMENTS_DIR
    
    @classmethod
    def get_source_files(cls) -> List[Dict[str, str]]:
        """
        Scan source directory and compare with collection.
        Returns: [{"name": "file.pdf", "status": "INGESTED" | "NEW"}]
        """
        source_dir = cls.get_source_dir()
        disk_files = set(os.listdir(source_dir))
        # Filter to PDF files only, exclude hidden files
        disk_files = {f for f in disk_files if f.endswith('.pdf') and not f.startswith('.')}
        
        ingested_files = set(cls.list_ingested_files())
        
        result = []
        for f in sorted(list(disk_files)):
            status = "INGESTED" if f in ingested_files else "NEW"
            result.append({"name": f, "status": status})
        
        return result
    
    @classmethod
    def list_ingested_files(cls) -> List[str]:
        """Return list of unique filenames in the collection."""
        collection = cls.get_collection()
        result = collection.get(include=["metadatas"])
        
        if not result['metadatas']:
            return []
        
        filenames = set()
        for meta in result['metadatas']:
            if meta and "original_name" in meta:
                filenames.add(meta["original_name"])
        
        return sorted(list(filenames))
    
    @classmethod
    def parse_pdf(cls, file_path: str) -> List[PdfChunk]:
        """
        Parse PDF into semantic chunks with metadata.
        """
        chunks = []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        try:
            reader = PdfReader(file_path)
            file_name = os.path.basename(file_path)
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    page_chunks = splitter.split_text(text.strip())
                    
                    for chunk_num, chunk_text in enumerate(page_chunks, 1):
                        if len(chunk_text.strip()) > 20:
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
    
    @classmethod
    def ingest_files(cls, filenames: List[str]) -> Dict[str, str]:
        """
        Ingest specified PDF files from source directory.
        Returns: {filename: "Success" | "Failed: reason"}
        """
        provider = EmbeddingFactory.get_provider()
        collection = cls.get_collection()
        results = {}
        
        for filename in filenames:
            file_path = os.path.join(QA_DOCUMENTS_DIR, filename)
            
            try:
                chunks = cls.parse_pdf(file_path)
                
                if not chunks:
                    results[filename] = "Empty file or parse error"
                    continue
                
                # Generate embeddings
                texts = [c.text for c in chunks]
                print(f"Generating embeddings for {len(texts)} chunks from {filename}...")
                embeddings = provider.embed_batch(texts)
                
                # Prepare data
                ids = [str(uuid.uuid4()) for _ in chunks]
                documents = texts
                metadatas = [c.metadata for c in chunks]
                
                # Upsert to collection
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                results[filename] = f"Success ({len(chunks)} chunks)"
                
            except Exception as e:
                results[filename] = f"Failed: {str(e)}"
        
        return results
    
    @classmethod
    def similarity_search_with_metadata(cls, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents and return with full metadata.
        Returns: [{"content": str, "metadata": dict}, ...]
        """
        provider = EmbeddingFactory.get_provider()
        collection = cls.get_collection()
        
        query_embedding = provider.embed_text(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        docs = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i] if results['metadatas'] else {}
            docs.append({
                "content": doc,
                "metadata": meta
            })
        
        return docs
    
    @classmethod
    async def ask(cls, question: str, model_slot: int = 1, k: int = 3) -> QAResponse:
        """
        Ask a question and get an answer with citations.
        """
        # 1. Retrieve relevant documents
        docs = cls.similarity_search_with_metadata(question, k=k)
        
        if not docs:
            return QAResponse(
                answer="I don't have any documents in my knowledge base yet. Please ingest some PDFs first.",
                citations=[],
                model_name="N/A"
            )
        
        # 2. Build context with citation markers
        context_parts = []
        citations = []
        
        for i, doc in enumerate(docs, 1):
            source = doc["metadata"].get("original_name", "Unknown")
            page = doc["metadata"].get("page_number", "?")
            content = doc["content"]
            
            context_parts.append(f"[{i}] Source: {source}, Page {page}\n{content}")
            
            # Build excerpt for citation
            excerpt = content[:80].replace("\n", " ").strip()
            if len(content) > 80:
                excerpt += "..."
            
            citations.append(Citation(
                index=i,
                source=source,
                page=page,
                excerpt=excerpt
            ))
        
        context_text = "\n\n".join(context_parts)
        
        # 3. Construct prompt
        qa_prompt = f"""Answer the question based on the following context.
Include citations in your answer using the format [1], [2], etc.
If you don't know the answer, say "I don't have enough information."

Context:
{context_text}

Question: {question}

Answer (with citations):"""
        
        # 4. Call LLM
        request = PromptRequest(
            user_query=qa_prompt,
            system_role="You are a Q&A assistant. Answer based on the provided context and cite your sources.",
            temperature=0.3
        )
        
        response = await prompt_engine.call_specific_model_by_slot(model_slot, request)
        
        # 5. Build response
        return QAResponse(
            answer=response.content,
            citations=citations,
            model_name=response.model_name
        )
    
    @classmethod
    def format_response_with_sources(cls, qa_response: QAResponse) -> str:
        """Format QA response with source legend."""
        output = qa_response.answer
        
        if qa_response.citations:
            output += "\n\n---\n**Sources:**\n"
            for c in qa_response.citations:
                output += f"\n[{c.index}] {c.source}, Page {c.page}\n    \"{c.excerpt}\"\n"
        
        return output
