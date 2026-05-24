import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- LangSmith Observability ---
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "finsight-ai")

if LANGSMITH_API_KEY and LANGSMITH_API_KEY != "your_langsmith_api_key_here":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "finsight_docs"

# --- Embeddings (free local model) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Sample docs directory ---
SAMPLE_DOCS_DIR = "./rag/sample_docs"

# --- Gemini model ---
GEMINI_MODEL = "gemini-flash-latest"  # Free tier: 1500 req/day
