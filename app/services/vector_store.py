import chromadb
import uuid
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.embedding_provider import EmbeddingFactory

class VectorStore:
    _client = None

    @classmethod
    def get_client(cls):
        if not cls._client:
            # Initialize persistent client
            cls._client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        return cls._client

    @classmethod
    def get_collection(cls):
        """
        Dynamically retrieve the collection for the ACTIVE provider.
        Collection Name: docs_{provider_name} (e.g., docs_google, docs_huggingface)
        """
        client = cls.get_client()
        provider = EmbeddingFactory.get_provider()
        collection_name = f"docs_{provider.provider_name}"
        
        return client.get_or_create_collection(name=collection_name)

    @classmethod
    def add_documents(cls, texts: List[str], filenames: List[str]):
        """
        Embeds and saves documents to the active collection.
        """
        if not texts:
            return

        provider = EmbeddingFactory.get_provider()
        collection = cls.get_collection()

        # Generate Embeddings
        print(f"Generating embeddings for {len(texts)} chunks using {provider.provider_name}...")
        embeddings = provider.embed_batch(texts)

        # Prepare Data
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        metadatas = [{"filename": fn} for fn in filenames]

        # Upsert to Chroma
        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(texts)} chunks to {collection.name}.")

    @classmethod
    def similarity_search(cls, query: str, k: int = 5) -> List[str]:
        """
        Embeds query and retrieves similar text chunks from active collection.
        """
        provider = EmbeddingFactory.get_provider()
        collection = cls.get_collection()

        query_embedding = provider.embed_text(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        if not results['documents']:
            return []
            
        return results['documents'][0]

    @classmethod
    def list_ingested_files(cls) -> List[str]:
        """
        Returns a list of unique filenames currently in the active collection.
        """
        collection = cls.get_collection()
        
        # Get all metadata (limit to a reasonable number, or fetch strictly what's needed)
        # Chroma doesn't support "distinct" query easily, so we fetch metadatas.
        # Warning: For massive datasets, this is inefficient. OK for local RAG.
        result = collection.get(include=["metadatas"])
        
        if not result['metadatas']:
            return []
            
        filenames = set()
        for meta in result['metadatas']:
            if meta and "filename" in meta:
                filenames.add(meta["filename"])
                
        return sorted(list(filenames))
