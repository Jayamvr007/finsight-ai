"""
RAG Ingestion Pipeline
Loads sample financial docs → Embeds → Stores in ChromaDB
Run this ONCE before starting the API server.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, SAMPLE_DOCS_DIR, EMBEDDING_MODEL

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader


def load_documents(docs_dir: str) -> list[dict]:
    """Load all .txt and .pdf files from sample_docs directory."""
    documents = []
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        content = ""
        
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        elif filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                content = "\n".join([page.page_content for page in pages])
            except Exception as e:
                print(f"Failed to load PDF {filename}: {e}")
                continue
        else:
            continue

        # Smart chunking with document-aware separators
        chunks = chunk_text(content)

        # Extract company name from filename for metadata enrichment
        company = extract_company_from_filename(filename)

        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{filename}_{i}",
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "company": company,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
    return documents


def extract_company_from_filename(filename: str) -> str:
    """Extract company name from document filename for metadata filtering."""
    name = filename.lower().replace(".txt", "").replace(".pdf", "")
    name = name.replace("_real", "").replace("_", " ").strip()

    company_map = {
        "reliance": "Reliance Industries",
        "tcs": "TCS",
        "hdfc bank": "HDFC Bank",
        "hdfc": "HDFC Bank",
        "infosys": "Infosys",
        "market overview": "Market Overview",
    }

    for key, value in company_map.items():
        if key in name:
            return value
    return name.title()


def chunk_text(text: str) -> list[str]:
    """
    Smart chunking using LangChain's RecursiveCharacterTextSplitter.
    Respects paragraph breaks, section headers, bullet points, and sentence boundaries
    instead of blindly cutting at character count.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",        # Paragraph breaks (highest priority)
            "\n===",       # Section headers in financial docs
            "\n---",       # Horizontal rules / separators
            "\n- ",        # Bullet points
            "\n• ",        # Unicode bullet points
            "\n",          # Line breaks
            ". ",          # Sentences
            " ",           # Words (last resort)
        ],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if c.strip()]


def ingest_documents():
    """Main ingestion function — loads docs and stores in ChromaDB."""
    print("🔄 Starting document ingestion...")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Use sentence-transformers embedding function (free, local)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Create or get collection
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Check if already populated
    existing_count = collection.count()
    if existing_count > 0:
        print(f"✅ Collection already has {existing_count} chunks. Skipping ingestion.")
        return collection

    # Load documents
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
    documents = load_documents(docs_dir)
    print(f"📄 Loaded {len(documents)} chunks from {docs_dir}")

    # Batch insert into ChromaDB
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        collection.add(
            ids=[doc["id"] for doc in batch],
            documents=[doc["text"] for doc in batch],
            metadatas=[doc["metadata"] for doc in batch]
        )
        print(f"  ✓ Ingested batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"✅ Ingestion complete! {collection.count()} total chunks in ChromaDB.")
    return collection


if __name__ == "__main__":
    ingest_documents()
