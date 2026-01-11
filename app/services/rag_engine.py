import os
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.services.vector_store import VectorStore
from app.utils.text_splitter import RecursiveCharacterTextSplitter
from app.services import prompt_engine
from app.models.schemas import PromptRequest, AIResponse

class RAGEngine:
    
    @staticmethod
    def get_source_files() -> List[Dict[str, str]]:
        """
        Scans source directory and compares with vector store.
        Returns: [{"name": "file.txt", "status": "INGESTED" | "NEW"}]
        """
        source_dir = settings.SOURCE_DOCUMENTS_DIR
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
            
        disk_files = set(os.listdir(source_dir))
        # Filter hidden files
        disk_files = {f for f in disk_files if not f.startswith(".")}
        
        ingested_files = set(VectorStore.list_ingested_files())
        
        result = []
        for f in sorted(list(disk_files)):
            status = "INGESTED" if f in ingested_files else "NEW"
            result.append({"name": f, "status": status})
            
        return result

    @staticmethod
    def ingest_files(filenames: List[str]) -> Dict[str, str]:
        """
        Ingests specified files from source directory.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        results = {}
        
        for filename in filenames:
            file_path = os.path.join(settings.SOURCE_DOCUMENTS_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                chunks = splitter.split_text(text)
                if chunks:
                    VectorStore.add_documents(chunks, [filename] * len(chunks))
                    results[filename] = "Success"
                else:
                    results[filename] = "Empty file or split error"
                    
            except UnicodeDecodeError:
                results[filename] = "Failed: Not a generic text file"
                # TODO: Implement PDF/Docx loader here later
            except Exception as e:
                results[filename] = f"Failed: {str(e)}"
                
        return results

    @staticmethod
    async def generate_rag_response(user_query: str, model_slot: int = 1) -> AIResponse:
        """
        Performs Retrieval Augmented Generation.
        1. Retrieve context
        2. Construct prompt
        3. Call LLM
        """
        # 1. Retrieve
        print("Retrieving context...")
        # Since our similarity_search returns 1 chunk (list of strings?), 
        # let's modify similarity_search signature later to return List[str] properly 
        # or handle current output. 
        # Assuming similarity_search returns List[str] of chunks.
        # Wait, previous implementation returned results['documents'][0] which is List[str] 
        # of the top k documents. So it returns List[str].
        
        context_chunks = VectorStore.similarity_search(user_query, k=3)
        
        if not context_chunks:
            context_text = "No relevant context found in knowledge base."
        else:
            context_text = "\n\n---\n\n".join(context_chunks)

        # 2. Construct Prompt
        rag_prompt = f"""
You are a helpful AI assistant. Answer the user's question based strictly on the provided context.
If the answer is not in the context, say "I don't have enough information in my knowledge base."

Context:
{context_text}

User Question: {user_query}

Answer:
"""
        # 3. Request
        request = PromptRequest(
            user_query=rag_prompt, # Using the constructed prompt as the query
            # We bypass system role or assume Call Model uses it. 
            # Ideally user_query is just the Q, but here we inject context.
            # Let's override user_query.
            system_role="You are a RAG assistant. Use the provided context.",
            temperature=0.3 # Lower temp for factual answers
        )
        
        return await prompt_engine.call_specific_model_by_slot(model_slot, request)
