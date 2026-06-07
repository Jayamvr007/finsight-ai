"""
Production Configuration
Loads all environment variables with sensible defaults and validation.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash-lite")

# ─────────────────────────────────────────
# LangSmith Observability
# ─────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "finsight-ai-prod")

if LANGSMITH_API_KEY and LANGSMITH_API_KEY != "your_langsmith_api_key_here":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# ─────────────────────────────────────────
# PostgreSQL (Supabase / self-hosted)
# ─────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/finsight"
)
# psycopg3 async-compatible URL
DATABASE_URL_ASYNC = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")

# ─────────────────────────────────────────
# Redis (Celery broker + result backend)
# ─────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ─────────────────────────────────────────
# ChromaDB (local fallback while migrating)
# ─────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = "finsight_docs"

# ─────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────
# Sample docs directory
# ─────────────────────────────────────────
SAMPLE_DOCS_DIR = "./rag/sample_docs"

# ─────────────────────────────────────────
# App Settings
# ─────────────────────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
