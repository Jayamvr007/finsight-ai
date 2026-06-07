"""
Celery Worker — Async Document Ingestion Pipeline
Handles PDF/TXT ingestion as background tasks so the web server is never blocked.

Usage:
    Start worker: celery -A tasks.worker worker --loglevel=info --concurrency=2
    Trigger task: from tasks.worker import ingest_document_task; ingest_document_task.delay(path)
"""

import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery import Celery
from config import REDIS_URL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Celery App
# ─────────────────────────────────────────

celery_app = Celery(
    "finsight_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Retry failed tasks with exponential backoff
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)


# ─────────────────────────────────────────
# Shared helpers (reused from ingest.py)
# ─────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """Smart recursive chunking respecting paragraph and sentence boundaries."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n===", "\n---", "\n- ", "\n• ", "\n", ". ", " "],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def _extract_text(filepath: str) -> str:
    """Extract plain text from .txt or .pdf files."""
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif filepath.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        pages = PyPDFLoader(filepath).load()
        return "\n".join(p.page_content for p in pages)
    return ""


def _get_chroma_collection():
    """Return a ChromaDB collection (initializes client if needed)."""
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────
# Celery Tasks
# ─────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.ingest_document",
    max_retries=3,
    default_retry_delay=10,
)
def ingest_document_task(self, filepath: str, company: str | None = None):
    """
    Background task: extract → chunk → embed → upsert into ChromaDB.

    Args:
        filepath: Absolute path to a .pdf or .txt document.
        company: Optional company label for metadata (auto-detected if None).
    
    Returns:
        dict with chunk_count and document source name.
    """
    filename = os.path.basename(filepath)
    logger.info(f"[Celery] Starting ingestion: {filename}")

    try:
        # 1. Extract text
        content = _extract_text(filepath)
        if not content.strip():
            logger.warning(f"[Celery] Empty content in {filename}, skipping.")
            return {"status": "skipped", "reason": "empty_content"}

        # 2. Smart chunking
        chunks = _chunk_text(content)

        # 3. Auto-detect company if not provided
        if not company:
            name = filename.lower().replace(".txt", "").replace(".pdf", "").replace("_", " ")
            company_map = {
                "reliance": "Reliance Industries",
                "tcs": "TCS",
                "hdfc": "HDFC Bank",
                "infosys": "Infosys",
            }
            company = next(
                (v for k, v in company_map.items() if k in name),
                name.title()
            )

        # 4. Build document records
        documents = [
            {
                "id": f"{filename}_{i}",
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "company": company,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
            for i, chunk in enumerate(chunks)
        ]

        # 5. Upsert into ChromaDB in batches (handles duplicates gracefully)
        collection = _get_chroma_collection()
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            collection.upsert(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
            )

        logger.info(f"[Celery] ✅ Ingested {len(chunks)} chunks from {filename}")
        return {"status": "success", "source": filename, "chunk_count": len(chunks)}

    except Exception as exc:
        logger.error(f"[Celery] ❌ Failed to ingest {filename}: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 10)


@celery_app.task(name="tasks.ingest_directory")
def ingest_directory_task(docs_dir: str):
    """Trigger individual ingestion tasks for every file in a directory."""
    count = 0
    for filename in os.listdir(docs_dir):
        if filename.endswith((".txt", ".pdf")):
            filepath = os.path.join(docs_dir, filename)
            ingest_document_task.delay(filepath)
            count += 1
    logger.info(f"[Celery] Queued {count} ingestion tasks from {docs_dir}")
    return {"queued": count}
