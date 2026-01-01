"""
API Routes Module

REST API endpoints for CropTAP RAG system.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    IngestRequest,
    IngestResponse,
    StatsResponse,
    HealthResponse
)
from app.config import settings
from llm.response_generator import get_response_generator
from retrieval.retriever import get_retriever
from retrieval.metadata_filter import VALID_CATEGORIES


router = APIRouter(tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Ask a question and get an AI-generated answer based on agricultural documents.
    
    The system retrieves relevant documents from the knowledge base and uses
    an LLM to generate a response grounded in the retrieved context.
    """
    # Validate category if provided
    if request.category and request.category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid options: {VALID_CATEGORIES}"
        )
    
    try:
        generator = get_response_generator()
        
        result = generator.answer(
            question=request.question,
            category=request.category,
            top_k=request.top_k
        )
        
        sources = [
            SourceDocument(
                content=s["content"],
                source=s["source"],
                category=s["category"],
                filename=s["filename"],
                score=s["score"]
            )
            for s in result["sources"]
        ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            question=result["question"],
            category_filter=result["category_filter"],
            documents_retrieved=result["documents_retrieved"]
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/categories")
async def list_categories() -> dict:
    """
    List available document categories for filtering.
    """
    return {
        "categories": VALID_CATEGORIES,
        "descriptions": {
            "crop_production_guide": "Crop production and farming guides",
            "crops_statistics": "Agricultural statistics and data",
            "planting_tips": "Planting tips and recommendations",
            "soil_data": "Soil properties and analysis data"
        }
    }


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get statistics about the document collection.
    """
    try:
        retriever = get_retriever()
        stats = retriever.get_collection_stats()
        
        if "error" in stats:
            raise HTTPException(
                status_code=503,
                detail=stats["error"]
            )
        
        return StatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(
    request: IngestRequest,
    background_tasks: BackgroundTasks
) -> IngestResponse:
    """
    Trigger document ingestion process.
    
    This will process all PDFs in the data directory and add them to the vector store.
    Note: This is a potentially long-running operation.
    """
    from ingestion.load_documents import ingest_documents
    
    try:
        data_dir = Path(request.data_dir) if request.data_dir else settings.data_dir
        
        if not data_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Data directory not found: {data_dir}"
            )
        
        # Run ingestion
        stats = ingest_documents(
            data_dir=data_dir,
            batch_size=request.batch_size
        )
        
        return IngestResponse(
            documents_processed=stats["documents_processed"],
            chunks_created=stats["chunks_created"],
            errors=stats["errors"],
            categories=stats["categories"],
            message="Ingestion completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during ingestion: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Detailed health check including component status.
    """
    from llm.llm_client import get_llm_client
    
    llm_client = get_llm_client()
    retriever = get_retriever()
    
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        service="croptap-rag",
        llm_configured=llm_client.is_configured(),
        vector_store_ready=retriever.collection is not None
    )


@router.get("/search")
async def search_documents(
    query: str,
    category: str = None,
    top_k: int = 5
) -> dict:
    """
    Search documents without generating an LLM response.
    
    Useful for debugging or when you only need retrieval results.
    """
    if category and category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid options: {VALID_CATEGORIES}"
        )
    
    try:
        retriever = get_retriever()
        results = retriever.search(
            query=query,
            top_k=top_k,
            category=category
        )
        
        return {
            "query": query,
            "category_filter": category,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching: {str(e)}"
        )
