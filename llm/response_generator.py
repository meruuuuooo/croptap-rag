"""
Response Generator Module

Orchestrates the full RAG pipeline: retrieve → build prompt → generate.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.llm_client import LLMClient, get_llm_client
from retrieval.retriever import Retriever, get_retriever
from prompt.prompt_builder import build_messages, format_sources


class ResponseGenerator:
    """
    Full RAG response generator.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[Retriever] = None
    ):
        """
        Initialize the response generator.
        
        Args:
            llm_client: LLM client instance.
            retriever: Retriever instance.
        """
        self.llm_client = llm_client or get_llm_client()
        self.retriever = retriever or get_retriever()
    
    def answer(
        self,
        question: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> dict:
        """
        Generate a RAG answer for a question.
        
        Args:
            question: User's question.
            category: Optional category filter.
            top_k: Number of documents to retrieve.
            
        Returns:
            Dictionary with 'answer', 'sources', and metadata.
        """
        logger.info(f"Processing question: {question[:100]}...")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.search(
            query=question,
            top_k=top_k,
            category=category
        )
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Build messages for LLM
        messages = build_messages(
            question=question,
            context_docs=retrieved_docs
        )
        
        # Step 3: Generate response
        answer = self.llm_client.generate(messages)
        
        # Step 4: Format sources
        sources = [
            {
                "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "source": doc["source"],
                "category": doc["category"],
                "filename": doc["filename"],
                "score": doc["score"]
            }
            for doc in retrieved_docs
        ]
        
        result = {
            "answer": answer,
            "sources": sources,
            "question": question,
            "category_filter": category,
            "documents_retrieved": len(retrieved_docs)
        }
        
        logger.info(f"Generated answer with {len(sources)} sources")
        
        return result
    
    def answer_with_threshold(
        self,
        question: str,
        threshold: float = 0.5,
        category: Optional[str] = None,
        top_k: int = 10
    ) -> dict:
        """
        Generate answer using only documents above relevance threshold.
        
        Args:
            question: User's question.
            threshold: Minimum relevance score.
            category: Optional category filter.
            top_k: Maximum documents to consider.
            
        Returns:
            Dictionary with 'answer', 'sources', and metadata.
        """
        # Retrieve with threshold
        retrieved_docs = self.retriever.search_with_threshold(
            query=question,
            threshold=threshold,
            max_results=top_k,
            category=category
        )
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find sufficiently relevant information to answer your question. "
                         "Could you try rephrasing or ask about a specific crop or farming topic?",
                "sources": [],
                "question": question,
                "category_filter": category,
                "documents_retrieved": 0,
                "threshold_used": threshold
            }
        
        # Build and generate
        messages = build_messages(
            question=question,
            context_docs=retrieved_docs
        )
        
        answer = self.llm_client.generate(messages)
        
        sources = [
            {
                "content": doc["content"][:200] + "...",
                "source": doc["source"],
                "category": doc["category"],
                "filename": doc["filename"],
                "score": doc["score"]
            }
            for doc in retrieved_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "category_filter": category,
            "documents_retrieved": len(retrieved_docs),
            "threshold_used": threshold
        }
    
    def is_ready(self) -> bool:
        """Check if the generator is ready (has both LLM and retriever)."""
        return (
            self.llm_client.is_configured() and 
            self.retriever.collection is not None
        )


# Singleton instance
_generator_instance: Optional[ResponseGenerator] = None


def get_response_generator() -> ResponseGenerator:
    """Get or create singleton response generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ResponseGenerator()
    return _generator_instance
