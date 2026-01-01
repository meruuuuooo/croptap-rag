"""
Text Cleaning Module

Normalizes and cleans extracted text from PDFs.
Removes artifacts, fixes encoding issues, and standardizes formatting.
"""

import re
import unicodedata
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text extracted from PDF.
        
    Returns:
        Cleaned and normalized text.
    """
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    
    # Fix common encoding issues
    text = fix_encoding_issues(text)
    
    # Remove headers and footers
    text = remove_headers_footers(text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def fix_encoding_issues(text: str) -> str:
    """
    Fix common encoding and character issues.
    
    Args:
        text: Input text.
        
    Returns:
        Text with fixed encoding.
    """
    # Common PDF extraction artifacts
    replacements = {
        '\ufb01': 'fi',
        '\ufb02': 'fl',
        '\ufb00': 'ff',
        '\ufb03': 'ffi',
        '\ufb04': 'ffl',
        '\u2019': "'",
        '\u2018': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2013': '-',
        '\u2014': '-',
        '\u2026': '...',
        '\xa0': ' ',  # Non-breaking space
        '\u200b': '',  # Zero-width space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def remove_headers_footers(text: str) -> str:
    """
    Remove common header/footer patterns from PDF text.
    
    Args:
        text: Input text.
        
    Returns:
        Text with headers/footers removed.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Patterns that typically indicate headers/footers
    header_footer_patterns = [
        r'^\s*page\s+\d+\s*(of\s+\d+)?\s*$',  # Page numbers
        r'^\s*\d+\s*$',  # Standalone numbers (page numbers)
        r'^\s*-\s*\d+\s*-\s*$',  # Page numbers with dashes
        r'^\s*©.*$',  # Copyright notices
        r'^\s*confidential\s*$',  # Confidentiality notices
    ]
    
    combined_pattern = '|'.join(header_footer_patterns)
    
    for line in lines:
        if not re.match(combined_pattern, line, re.IGNORECASE):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace and line breaks.
    
    Args:
        text: Input text.
        
    Returns:
        Text with normalized whitespace.
    """
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace from lines
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    
    return text


def extract_sections(text: str) -> list[dict]:
    """
    Attempt to extract sections based on common heading patterns.
    
    Args:
        text: Input text.
        
    Returns:
        List of dictionaries with 'heading' and 'content' keys.
    """
    # Common heading patterns in agricultural documents
    heading_pattern = r'^(?:(?:\d+\.?\s*)+|[A-Z][A-Z\s]+:?|[IVX]+\.)(.+)$'
    
    sections = []
    current_section = {"heading": "Introduction", "content": []}
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line looks like a heading
        if re.match(heading_pattern, line) and len(line) < 100:
            if current_section["content"]:
                current_section["content"] = '\n'.join(current_section["content"])
                sections.append(current_section)
            current_section = {"heading": line, "content": []}
        else:
            current_section["content"].append(line)
    
    # Add last section
    if current_section["content"]:
        current_section["content"] = '\n'.join(current_section["content"])
        sections.append(current_section)
    
    return sections


def clean_for_embedding(text: str) -> str:
    """
    Prepare text specifically for embedding generation.
    Removes elements that don't contribute to semantic meaning.
    
    Args:
        text: Cleaned text.
        
    Returns:
        Text optimized for embedding.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{4,}', '...', text)
    text = re.sub(r'[-]{4,}', '---', text)
    
    # Normalize whitespace again
    text = normalize_whitespace(text)
    
    return text.strip()
