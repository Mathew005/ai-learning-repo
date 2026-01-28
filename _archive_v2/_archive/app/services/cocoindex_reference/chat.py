"""
RAG Chat with Citations - REFERENCE IMPLEMENTATION

Original source: project/rag_app/chat.py
Archived for future implementation and reference.

This file shows how to create a RAG chain with LangChain that includes citations.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    """
    Parse model specification into provider and model name.
    
    Examples:
        "ollama/gemma:2b" -> ("ollama", "gemma:2b")
        "google/gemini-2.0-flash" -> ("google", "gemini-2.0-flash")
    """
    if "/" not in model_spec:
        raise ValueError(f"Invalid model format: {model_spec}. Expected: provider/model")
    
    provider, model = model_spec.split("/", 1)
    return provider.lower(), model


def get_llm(model_spec: str, ollama_base_url: str, google_api_key: str = None):
    """Get LLM based on model specification."""
    provider, model = parse_model_spec(model_spec)
    
    if provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=model,
            base_url=ollama_base_url
        )
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, google")


def get_vectorstore(chroma_path: str, collection_name: str, embedding_model: str, ollama_base_url: str):
    """Connect to existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_path)
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_base_url
    )
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )


def format_docs_with_citations(docs):
    """Format retrieved documents with source citations."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("original_name", "Unknown")
        page = doc.metadata.get("page_number", "?")
        content = doc.page_content
        formatted.append(f"[{i}] Source: {source}, Page {page}\n{content}")
    return "\n\n".join(formatted)


def create_rag_chain(retriever, llm):
    """Create a RAG chain with citations."""
    template = """Answer the question based on the following context. 
Include citations in your answer using the format [1], [2], etc.
If you don't know the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer (with citations):"""

    prompt = ChatPromptTemplate.from_template(template)
    
    def process_with_sources(inputs):
        """Process query and append source details."""
        docs = retriever.invoke(inputs["question"])
        context = format_docs_with_citations(docs)
        
        # Build source legend with excerpts
        sources = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("original_name", "Unknown")
            page = doc.metadata.get("page_number", "?")
            # Get first 80 chars as excerpt
            excerpt = doc.page_content[:80].replace("\n", " ").strip()
            if len(doc.page_content) > 80:
                excerpt += "..."
            sources.append(f"[{i}] {source}, Page {page}\n    \"{excerpt}\"")

        
        # Get LLM response
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": inputs["question"]})
        
        # Append source legend
        source_legend = "\n\n---\n**Sources:**\n" + "\n\n".join(sources)
        return answer + source_legend
    
    return process_with_sources


def chat(question: str, retriever, llm) -> str:
    """
    Ask a question and get an answer with citations.
    
    Args:
        question: The user's question
        retriever: The document retriever
        llm: The language model
    """
    chain_fn = create_rag_chain(retriever, llm)
    return chain_fn({"question": question})
