import streamlit as st
import importlib

# Configure Page
st.set_page_config(
    page_title="AI Learning Repo",
    page_icon="🧠",
    layout="wide"
)

# --- Page Functions ---

def home_page():
    st.title("🧠 AI Learning Roadmap")
    st.markdown("""
    ### Welcome to the AI Learning Repo
    
    This dashboard integrates all learning phases into a single application.
    Select a phase from the sidebar to interact with it.
    
    *   **Architecture**: Shared `litellm` core.
    *   **Configuration**: Centralized `.env`.
    """)

def direct_chat_page():
    try:
        # Dynamic import to handle the module
        chat_interface = importlib.import_module("01_Direct_Chat.chat_interface")
        chat_interface.render()
    except ImportError as e:
        st.error(f"Failed to load Phase 1: {e}")

def chain_of_thought_page():
    try:
        chain_interface = importlib.import_module("02_Chain_Of_Thought.chain_interface")
        chain_interface.render()
    except ImportError as e:
        st.error(f"Failed to load Phase 2: {e}")

def basic_rag_page():
    st.title("03. Basic RAG")
    st.info("Phase 3 Placeholder. Implementation coming soon.")

# --- Navigation Setup ---

pg = st.navigation({
    "Overview": [
        st.Page(home_page, title="Home", icon="🏠"),
    ],
    "Learning Phases": [
        st.Page(direct_chat_page, title="01. Direct Chat", icon="💬"),
        st.Page(chain_of_thought_page, title="02. Chain of Thought", icon="🔗"),
        st.Page(basic_rag_page, title="03. Basic RAG", icon="📚"),
    ]
})

pg.run()
