"""
PDF to Text Extraction Module

Extracts text content from PDF files using PyMuPDF (fitz).
Handles batch processing with metadata preservation.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Generator
from loguru import logger


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract all text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Extracted text content as a single string.
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            if page_text.strip():
                text_content.append(page_text)
        
        doc.close()
        
        full_text = "\n\n".join(text_content)
        logger.debug(f"Extracted {len(full_text)} characters from {pdf_path.name}")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        raise


def extract_all_pdfs(
    directory: str | Path,
    recursive: bool = True
) -> Generator[dict, None, None]:
    """
    Extract text from all PDFs in a directory.
    
    Args:
        directory: Directory containing PDF files.
        recursive: Whether to search subdirectories.
        
    Yields:
        Dictionary with 'text', 'source', 'category', and 'filename' keys.
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(directory.glob(pattern))
    
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    
    for pdf_path in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_path)
            
            # Extract category from parent directory name
            category = pdf_path.parent.name if pdf_path.parent != directory else "uncategorized"
            
            yield {
                "text": text,
                "source": str(pdf_path),
                "category": category,
                "filename": pdf_path.name
            }
            
        except Exception as e:
            logger.warning(f"Skipping {pdf_path.name}: {e}")
            continue


def get_pdf_metadata(pdf_path: str | Path) -> dict:
    """
    Extract metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Dictionary containing PDF metadata.
    """
    pdf_path = Path(pdf_path)
    
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        page_count = len(doc)
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "page_count": page_count,
            "filename": pdf_path.name
        }
        
    except Exception as e:
        logger.error(f"Error reading metadata from {pdf_path}: {e}")
        return {}
