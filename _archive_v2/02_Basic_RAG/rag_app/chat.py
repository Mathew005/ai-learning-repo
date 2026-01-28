"""
RAG Chat Application using LangChain + ChromaDB + Configurable LLM

Configuration via config.py and .env file.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb

# Import configuration
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from config import Config, Models


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


def get_llm(model_spec: str = None):
    """Get LLM based on model specification."""
    spec = model_spec or Config.get_model()
    provider, model = parse_model_spec(spec)
    
    if provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=model,
            base_url=Config.OLLAMA_BASE_URL
        )
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in .env file")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=Config.GOOGLE_API_KEY
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, google")


def get_vectorstore():
    """Connect to existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
    embeddings = OllamaEmbeddings(
        model=Config.EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL
    )
    return Chroma(
        client=client,
        collection_name=Config.COLLECTION_NAME,
        embedding_function=embeddings
    )


def get_retriever():
    """Get a retriever that returns top-k relevant documents."""
    return get_vectorstore().as_retriever(search_kwargs={"k": Config.TOP_K})


def format_docs_with_citations(docs):
    """Format retrieved documents with source citations."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("original_name", "Unknown")
        page = doc.metadata.get("page_number", "?")
        content = doc.page_content
        formatted.append(f"[{i}] Source: {source}, Page {page}\n{content}")
    return "\n\n".join(formatted)


def create_rag_chain(model_spec: str = None):
    """Create a RAG chain with citations."""
    retriever = get_retriever()
    llm = get_llm(model_spec)
    
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


def chat(question: str, model_spec: str = None) -> str:
    """
    Ask a question and get an answer with citations.
    
    Args:
        question: The user's question
        model_spec: Optional model override (e.g., Models.GOOGLE_GEMINI)
    """
    chain_fn = create_rag_chain(model_spec)
    return chain_fn({"question": question})


def interactive_chat():
    """Run an interactive chat session."""
    model = Config.get_model()
    provider, model_name = parse_model_spec(model)
    
    print("=" * 60)
    print(f"🤖 Multi-Source RAG Chat")
    print(f"   Provider: {provider}")
    print(f"   Model: {model_name}")
    print("=" * 60)
    print("Ask questions about your documents. Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
            
        if not question:
            continue
            
        print("\n🔍 Searching documents...")
        try:
            answer = chat(question)
            print(f"\n📚 Answer:\n{answer}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    interactive_chat()
