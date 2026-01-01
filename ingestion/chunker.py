"""
Document Chunking Module

Splits documents into overlapping chunks for embedding.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
"""

import sys
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings


def create_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with the specified parameters.
    
    Args:
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )


def chunk_document(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> list[str]:
    """
    Split a document into chunks.
    
    Args:
        text: Document text to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []
    
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    
    logger.debug(f"Split document into {len(chunks)} chunks")
    
    return chunks


def chunk_with_metadata(
    document: dict,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> list[dict]:
    """
    Split a document into chunks while preserving metadata.
    
    Args:
        document: Dictionary with 'text' and metadata keys.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of chunk dictionaries with preserved metadata.
    """
    text = document.get("text", "")
    
    if not text:
        return []
    
    chunks = chunk_document(text, chunk_size, chunk_overlap)
    
    # Preserve metadata for each chunk
    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_doc = {
            "text": chunk_text,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source": document.get("source", "unknown"),
            "category": document.get("category", "uncategorized"),
            "filename": document.get("filename", "unknown")
        }
        result.append(chunk_doc)
    
    return result


def chunk_documents(
    documents: list[dict],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> list[dict]:
    """
    Chunk multiple documents with metadata preservation.
    
    Args:
        documents: List of document dictionaries.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of all chunks from all documents.
    """
    all_chunks = []
    
    for doc in documents:
        doc_chunks = chunk_with_metadata(doc, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    return all_chunks


def estimate_chunk_count(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> int:
    """
    Estimate the number of chunks without actually splitting.
    
    Args:
        text: Document text.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        Estimated chunk count.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    if len(text) <= chunk_size:
        return 1
    
    effective_size = chunk_size - chunk_overlap
    return max(1, (len(text) - chunk_overlap) // effective_size + 1)
