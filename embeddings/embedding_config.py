"""
Embedding Configuration Module

Configuration constants for the embedding model.
"""

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Alternative models (for reference)
ALTERNATIVE_MODELS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_sequence_length": 256,
        "description": "Fast, lightweight model suitable for semantic search"
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_sequence_length": 384,
        "description": "Higher quality, slower model"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "max_sequence_length": 128,
        "description": "Multilingual support"
    }
}

# Batch processing
DEFAULT_BATCH_SIZE = 32
MAX_BATCH_SIZE = 128
