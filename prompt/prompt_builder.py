"""
Prompt Builder Module

Constructs prompts for LLM using retrieved context.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt.prompt_template import (
    SYSTEM_PROMPT,
    RAG_TEMPLATE,
    NO_CONTEXT_TEMPLATE,
    MULTI_SOURCE_TEMPLATE
)


def format_context(documents: list[dict], max_chars: int = 4000) -> str:
    """
    Format retrieved documents into a context string.
    
    Args:
        documents: List of retrieved documents.
        max_chars: Maximum characters for context.
        
    Returns:
        Formatted context string.
    """
    if not documents:
        return ""
    
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(documents):
        content = doc.get("content", "")
        source = doc.get("filename", "Unknown source")
        
        entry = f"[Source: {source}]\n{content}"
        
        if current_length + len(entry) > max_chars:
            # Truncate last entry if needed
            remaining = max_chars - current_length - 50
            if remaining > 100:
                entry = entry[:remaining] + "..."
                context_parts.append(entry)
            break
        
        context_parts.append(entry)
        current_length += len(entry) + 2  # +2 for separator
    
    return "\n\n---\n\n".join(context_parts)


def format_sources(documents: list[dict]) -> str:
    """
    Format source citations.
    
    Args:
        documents: List of retrieved documents.
        
    Returns:
        Formatted source list.
    """
    if not documents:
        return "No sources found."
    
    sources = []
    seen = set()
    
    for doc in documents:
        filename = doc.get("filename", "Unknown")
        category = doc.get("category", "unknown")
        
        key = f"{filename}|{category}"
        if key not in seen:
            seen.add(key)
            sources.append(f"- {filename} ({category})")
    
    return "\n".join(sources)


def build_messages(
    question: str,
    context_docs: list[dict],
    include_sources: bool = True
) -> list[dict]:
    """
    Build OpenAI message format for RAG.
    
    Args:
        question: User's question.
        context_docs: Retrieved context documents.
        include_sources: Whether to include source citations.
        
    Returns:
        List of message dictionaries for OpenAI API.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    if not context_docs:
        # No context found
        user_content = NO_CONTEXT_TEMPLATE.format(question=question)
    else:
        # Build context-grounded prompt
        context = format_context(context_docs)
        
        if include_sources and len(set(d.get("filename") for d in context_docs)) > 1:
            # Multiple sources - use multi-source template
            sources = format_sources(context_docs)
            user_content = MULTI_SOURCE_TEMPLATE.format(
                sources=sources,
                context=context,
                question=question
            )
        else:
            # Standard RAG template
            user_content = RAG_TEMPLATE.format(
                context=context,
                question=question
            )
    
    messages.append({"role": "user", "content": user_content})
    
    return messages


def build_simple_prompt(question: str, context: str) -> str:
    """
    Build a simple prompt string (for non-chat models).
    
    Args:
        question: User's question.
        context: Context string.
        
    Returns:
        Complete prompt string.
    """
    return RAG_TEMPLATE.format(context=context, question=question)
