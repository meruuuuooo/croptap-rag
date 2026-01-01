"""
Metadata Filter Module

Builds ChromaDB filter clauses for category and source filtering.
"""

from typing import Optional


# Valid document categories
VALID_CATEGORIES = [
    "crop_production_guide",
    "crops_statistics",
    "planting_tips",
    "soil_data"
]


def build_filter(
    category: Optional[str] = None,
    source: Optional[str] = None,
    filename: Optional[str] = None
) -> Optional[dict]:
    """
    Build a ChromaDB where clause for filtering.
    
    Args:
        category: Filter by document category.
        source: Filter by source path (contains match).
        filename: Filter by filename (contains match).
        
    Returns:
        ChromaDB where clause dictionary or None if no filters.
    """
    conditions = []
    
    if category:
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: {category}. "
                f"Valid categories: {VALID_CATEGORIES}"
            )
        conditions.append({"category": {"$eq": category}})
    
    if source:
        conditions.append({"source": {"$contains": source}})
    
    if filename:
        conditions.append({"filename": {"$contains": filename}})
    
    if not conditions:
        return None
    
    if len(conditions) == 1:
        return conditions[0]
    
    return {"$and": conditions}


def validate_category(category: str) -> bool:
    """
    Check if a category is valid.
    
    Args:
        category: Category name to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    return category in VALID_CATEGORIES


def get_category_description(category: str) -> str:
    """
    Get a human-readable description for a category.
    
    Args:
        category: Category name.
        
    Returns:
        Description string.
    """
    descriptions = {
        "crop_production_guide": "Crop production and farming guides",
        "crops_statistics": "Agricultural statistics and data",
        "planting_tips": "Planting tips and recommendations",
        "soil_data": "Soil properties and analysis data"
    }
    return descriptions.get(category, "Unknown category")
