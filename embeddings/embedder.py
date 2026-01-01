"""
Text Embedder Module

Converts text to vector embeddings using sentence-transformers.
"""

import sys
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from embeddings.embedding_config import DEFAULT_BATCH_SIZE


class Embedder:
    """
    Text embedding generator using sentence-transformers.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedder with the specified model.
        
        Args:
            model_name: Name of the sentence-transformer model.
        """
        self.model_name = model_name or settings.embedding_model
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Get embedding dimension from model
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed.
            
        Returns:
            List of floats representing the embedding vector.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once.
            show_progress: Whether to show a progress bar.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        
        # Filter empty texts and track indices
        valid_indices = []
        valid_texts = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
        
        if not valid_texts:
            return [[0.0] * self.dimension] * len(texts)
        
        # Generate embeddings
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Reconstruct full list with zero vectors for empty texts
        result = [[0.0] * self.dimension] * len(texts)
        for i, embedding in zip(valid_indices, embeddings):
            result[i] = embedding.tolist()
        
        return result
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Cosine similarity score (0-1).
        """
        from sentence_transformers import util
        
        emb1 = self.model.encode(text1, convert_to_numpy=True)
        emb2 = self.model.encode(text2, convert_to_numpy=True)
        
        return float(util.cos_sim(emb1, emb2)[0][0])


# Singleton instance for convenience
_embedder_instance: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Get or create singleton embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
