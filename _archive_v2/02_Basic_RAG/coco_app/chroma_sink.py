import chromadb
import uuid
from typing import List, Dict, Any

# Global client to avoid re-initializing per batch if possible, 
# or we can init per call. For local persist, it's fine.
db_path = "./chroma_db"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="coco_collection")

def write_to_chroma(batch: List[Dict[str, Any]]) -> List[str]:
    """
    Writes a batch of records to ChromaDB.
    Expected format: 
    {
        "content": str,
        "embedding": list[float],
        "metadata": dict
    }
    """
    if not batch:
        return []
    
    ids = [str(uuid.uuid4()) for _ in batch]
    documents = [item["content"] for item in batch]
    embeddings = [item["embedding"] for item in batch]
    metadatas = [item["metadata"] for item in batch]
    
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"💾 Persisted {len(batch)} chunks to ChromaDB.")
    return ids
