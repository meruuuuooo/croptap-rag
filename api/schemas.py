"""
API Schemas Module

Pydantic models for request/response validation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask about agricultural topics"
    )
    category: Optional[str] = Field(
        None,
        description="Optional category filter (crop_production_guide, crops_statistics, planting_tips, soil_data)"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of relevant documents to retrieve"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I grow bananas in the Philippines?",
                "category": "crop_production_guide",
                "top_k": 5
            }
        }


class SourceDocument(BaseModel):
    """Model for source document citations."""
    
    content: str = Field(..., description="Excerpt from the source document")
    source: str = Field(..., description="Full path to source file")
    category: str = Field(..., description="Document category")
    filename: str = Field(..., description="Source filename")
    score: float = Field(..., description="Relevance score (0-1)")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    
    answer: str = Field(..., description="Generated answer to the question")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used to generate the answer"
    )
    question: str = Field(..., description="Original question")
    category_filter: Optional[str] = Field(None, description="Category filter applied")
    documents_retrieved: int = Field(..., description="Number of documents retrieved")


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    
    data_dir: Optional[str] = Field(
        None,
        description="Optional custom data directory path"
    )
    batch_size: int = Field(
        100,
        ge=10,
        le=500,
        description="Batch size for embedding generation"
    )


class IngestResponse(BaseModel):
    """Response model for ingestion results."""
    
    documents_processed: int
    chunks_created: int
    errors: int
    categories: dict[str, int]
    message: str


class StatsResponse(BaseModel):
    """Response model for collection statistics."""
    
    total_chunks: int
    collection_name: str
    categories_sample: dict[str, int]
    available_categories: list[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    version: str
    service: str
    llm_configured: bool
    vector_store_ready: bool
