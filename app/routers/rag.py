from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore
from app.models.schemas import AIResponse

router = APIRouter(prefix="/rag", tags=["RAG"])

class RAGQueryRequest(BaseModel):
    query: str
    model_slot: int = 1

class IngestRequest(BaseModel):
    filenames: Optional[List[str]] = None  # If None, ingest all NEW files

@router.post("/query", response_model=AIResponse)
async def query_rag(request: RAGQueryRequest):
    """
    Ask a question to the Knowledge Base.
    """
    try:
        response = await RAGEngine.generate_rag_response(
            user_query=request.query,
            model_slot=request.model_slot
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents from the source directory.
    If 'filenames' is provided, only those are ingested.
    Otherwise, scans for all valid files.
    """
    try:
        if request.filenames:
            # Ingest specific list
            files_to_ingest = request.filenames
        else:
            # Auto-discover
            source_files = RAGEngine.get_source_files()
            # Default to ingesting everything (or just NEW ones? Let's just pass them all to ingest_files)
            # RAGEngine.ingest_files expects a list of filenames.
            files_to_ingest = [f['name'] for f in source_files]
            
        if not files_to_ingest:
             return {"message": "No files found to ingest.", "results": {}}

        results = RAGEngine.ingest_files(files_to_ingest)
        return {"message": "Ingestion complete", "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """
    Get the status of files in the current provider's collection.
    """
    try:
        files = RAGEngine.get_source_files()
        collection_name = VectorStore.get_collection().name
        
        ingested_count = sum(1 for f in files if f['status'] == "INGESTED")
        total_count = len(files)
        
        return {
            "collection": collection_name,
            "counts": {
                "ingested": ingested_count,
                "total": total_count
            },
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
