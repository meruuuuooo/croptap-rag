"""
Retriever Module

Semantic search over the vector store using ChromaDB.
"""

import sys
from pathlib import Path
from typing import Optional

import chromadb
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from embeddings.embedder import Embedder, get_embedder
from retrieval.metadata_filter import build_filter, VALID_CATEGORIES


class Retriever:
    """
    Semantic search retriever using ChromaDB.
    """
    
    def __init__(
        self,
        chroma_client: Optional[chromadb.PersistentClient] = None,
        embedder: Optional[Embedder] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            chroma_client: ChromaDB client instance.
            embedder: Embedder instance for query embedding.
        """
        self.client = chroma_client or chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )
        self.embedder = embedder or get_embedder()
        
        try:
            self.collection = self.client.get_collection(
                name=settings.collection_name
            )
            logger.info(
                f"Connected to collection '{settings.collection_name}' "
                f"with {self.collection.count()} documents"
            )
        except ValueError:
            logger.warning(
                f"Collection '{settings.collection_name}' not found. "
                "Run document ingestion first."
            )
            self.collection = None
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None
    ) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            category: Optional category filter.
            
        Returns:
            List of dictionaries with 'content', 'source', 'category', 'score'.
        """
        if self.collection is None:
            logger.error("No collection available. Run document ingestion first.")
            return []
        
        top_k = top_k or settings.default_top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Build filter if category specified
        where_filter = build_filter(category=category) if category else None
        
        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                # Lower distance = more similar, so we invert
                score = max(0, 1 - (distance / 2))
                
                formatted_results.append({
                    "content": doc,
                    "source": metadata.get("source", "unknown"),
                    "category": metadata.get("category", "uncategorized"),
                    "filename": metadata.get("filename", "unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "total_chunks": metadata.get("total_chunks", 1),
                    "score": round(score, 4)
                })
        
        logger.debug(f"Found {len(formatted_results)} results for query: {query[:50]}...")
        
        return formatted_results
    
    def search_with_threshold(
        self,
        query: str,
        threshold: float = 0.5,
        max_results: int = 10,
        category: Optional[str] = None
    ) -> list[dict]:
        """
        Search with a minimum relevance threshold.
        
        Args:
            query: Search query text.
            threshold: Minimum similarity score (0-1).
            max_results: Maximum results to consider.
            category: Optional category filter.
            
        Returns:
            List of results above the threshold.
        """
        results = self.search(query, top_k=max_results, category=category)
        return [r for r in results if r["score"] >= threshold]
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        if self.collection is None:
            return {"error": "No collection available"}
        
        count = self.collection.count()
        
        # Get category distribution (sample-based for large collections)
        sample_size = min(1000, count)
        sample = self.collection.peek(limit=sample_size)
        
        categories = {}
        if sample["metadatas"]:
            for meta in sample["metadatas"]:
                cat = meta.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_chunks": count,
            "collection_name": settings.collection_name,
            "categories_sample": categories,
            "available_categories": VALID_CATEGORIES
        }


# Singleton instance
_retriever_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get or create singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
