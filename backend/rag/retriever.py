"""
RAG Retriever
Queries ChromaDB using semantic similarity search.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

import chromadb
from chromadb.utils import embedding_functions


_client = None
_collection = None


def get_collection():
    """Lazy singleton — reuse ChromaDB client across requests."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def retrieve(query: str, n_results: int = 5, source_filter: str = None) -> list[dict]:
    """
    Semantic similarity search over ChromaDB.

    Args:
        query: The search query string
        n_results: Number of top chunks to return
        source_filter: Optional — filter by filename (e.g., 'reliance.txt')

    Returns:
        List of dicts: [{text, source, score}]
    """
    collection = get_collection()

    where_filter = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "relevance_score": round(1 - dist, 4)  # Convert cosine distance to similarity
            })

    return retrieved


def retrieve_for_company(company_name: str, query: str, n_results: int = 4) -> list[dict]:
    """
    Company-focused retrieval: tries company-specific docs first,
    falls back to general market docs.
    """
    # Map common names to filenames
    company_map = {
        "reliance": "reliance.txt",
        "ril": "reliance.txt",
        "tcs": "tcs.txt",
        "tata consultancy": "tcs.txt",
        "hdfc": "hdfc_bank.txt",
        "hdfc bank": "hdfc_bank.txt",
        "infosys": "infosys_real.pdf",
        "infy": "infosys_real.pdf",
    }

    # Find matching file
    source_file = None
    for key, filename in company_map.items():
        if key in company_name.lower():
            source_file = filename
            break

    # Primary retrieval (company-specific)
    results = retrieve(query, n_results=n_results, source_filter=source_file)

    # Always add market overview context
    market_results = retrieve(query, n_results=2, source_filter="market_overview.txt")

    # Combine — deduplicate by text
    seen = set()
    combined = []
    for r in results + market_results:
        if r["text"] not in seen:
            seen.add(r["text"])
            combined.append(r)

    return combined
