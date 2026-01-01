"""
CropTAP RAG Configuration Module

Centralized configuration management using pydantic-settings.
Loads settings from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434/v1"
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data" / "raw"
    processed_dir: Path = base_dir / "data" / "processed"
    chroma_persist_dir: Path = base_dir / "vector_store"
    logs_dir: Path = base_dir / "logs"
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # LLM Configuration (Ollama)
    llm_model: str = "llama3.2"  # or qwen2.5, mistral, gemma2, etc.
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    
    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval Configuration
    default_top_k: int = 5
    
    # ChromaDB Configuration
    collection_name: str = "croptap_docs"
    
    # API Configuration
    api_title: str = "CropTAP RAG API"
    api_version: str = "1.0.0"
    api_description: str = "Agricultural Knowledge Retrieval API"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
