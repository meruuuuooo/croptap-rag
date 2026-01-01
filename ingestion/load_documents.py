"""
Document Loading Module

Orchestrates the full document ingestion pipeline:
PDF extraction → Text cleaning → Chunking → Embedding → Vector store.
"""

import sys
from pathlib import Path
from typing import Optional

import chromadb
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from ingestion.pdf_to_text import extract_all_pdfs
from ingestion.text_cleaner import clean_text, clean_for_embedding
from ingestion.chunker import chunk_with_metadata
from embeddings.embedder import Embedder


def get_category_from_path(file_path: str | Path) -> str:
    """
    Extract category from file path.
    
    Args:
        file_path: Path to the document.
        
    Returns:
        Category name from parent directory.
    """
    path = Path(file_path)
    
    # Get the parent directory name relative to data/raw
    parts = path.parts
    if "raw" in parts:
        raw_idx = parts.index("raw")
        if len(parts) > raw_idx + 1:
            return parts[raw_idx + 1]
    
    return path.parent.name


def process_document(document: dict) -> list[dict]:
    """
    Process a single document through the cleaning and chunking pipeline.
    
    Args:
        document: Dictionary with 'text' and metadata.
        
    Returns:
        List of processed chunks.
    """
    # Clean the text
    cleaned_text = clean_text(document["text"])
    embedding_text = clean_for_embedding(cleaned_text)
    
    # Update document with cleaned text
    processed_doc = {
        **document,
        "text": embedding_text
    }
    
    # Chunk the document
    chunks = chunk_with_metadata(processed_doc)
    
    return chunks


def ingest_documents(
    data_dir: Optional[str | Path] = None,
    batch_size: int = 100
) -> dict:
    """
    Full ingestion pipeline: extract → clean → chunk → embed → store.
    
    Args:
        data_dir: Directory containing PDF documents.
        batch_size: Number of chunks to embed at once.
        
    Returns:
        Dictionary with ingestion statistics.
    """
    data_dir = Path(data_dir) if data_dir else settings.data_dir
    
    logger.info(f"Starting document ingestion from {data_dir}")
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(name=settings.collection_name)
        logger.info(f"Deleted existing collection: {settings.collection_name}")
    except Exception:
        pass  # Collection doesn't exist, that's fine
    
    # Create new collection
    collection = chroma_client.create_collection(
        name=settings.collection_name,
        metadata={"description": "CropTAP agricultural documents"}
    )
    
    # Initialize embedder
    embedder = Embedder()
    
    # Statistics
    stats = {
        "documents_processed": 0,
        "chunks_created": 0,
        "errors": 0,
        "categories": {}
    }
    
    # Process all PDFs
    all_chunks = []
    
    for doc in extract_all_pdfs(data_dir):
        try:
            chunks = process_document(doc)
            all_chunks.extend(chunks)
            
            stats["documents_processed"] += 1
            
            # Track category stats
            category = doc.get("category", "uncategorized")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            if stats["documents_processed"] % 50 == 0:
                logger.info(f"Processed {stats['documents_processed']} documents...")
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            stats["errors"] += 1
            continue
    
    stats["chunks_created"] = len(all_chunks)
    logger.info(f"Created {len(all_chunks)} chunks from {stats['documents_processed']} documents")
    
    # Embed and store in batches
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        
        texts = [chunk["text"] for chunk in batch]
        embeddings = embedder.embed_batch(texts)
        
        # Prepare for ChromaDB
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas = [
            {
                "source": chunk["source"],
                "category": chunk["category"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"]
            }
            for chunk in batch
        ]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.debug(f"Added batch {i // batch_size + 1} to vector store")
    
    logger.info(f"Ingestion complete. Stats: {stats}")
    
    return stats


def add_document(
    pdf_path: str | Path,
    category: Optional[str] = None
) -> dict:
    """
    Add a single document to the vector store.
    
    Args:
        pdf_path: Path to the PDF file.
        category: Optional category override.
        
    Returns:
        Dictionary with addition statistics.
    """
    from ingestion.pdf_to_text import extract_text_from_pdf
    
    pdf_path = Path(pdf_path)
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # Create document dict
    doc = {
        "text": text,
        "source": str(pdf_path),
        "category": category or get_category_from_path(pdf_path),
        "filename": pdf_path.name
    }
    
    # Process document
    chunks = process_document(doc)
    
    # Initialize ChromaDB client and embedder
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = chroma_client.get_collection(name=settings.collection_name)
    embedder = Embedder()
    
    # Get current count for ID generation
    current_count = collection.count()
    
    # Embed and store
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_batch(texts)
    
    ids = [f"chunk_{current_count + i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": chunk["source"],
            "category": chunk["category"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"]
        }
        for chunk in chunks
    ]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    logger.info(f"Added {len(chunks)} chunks from {pdf_path.name}")
    
    return {
        "filename": pdf_path.name,
        "chunks_added": len(chunks),
        "category": doc["category"]
    }


if __name__ == "__main__":
    # Run ingestion when executed directly
    stats = ingest_documents()
    print(f"\nIngestion complete!")
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Errors: {stats['errors']}")
    print(f"Categories: {stats['categories']}")
