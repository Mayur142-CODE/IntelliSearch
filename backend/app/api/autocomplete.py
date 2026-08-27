"""
Autocomplete API Endpoint

GET /autocomplete?q=<partial_query>

Returns complete-query dynamic suggestions from the CatalogVocabulary singleton
and PostgreSQL product catalog. No hardcoded brands, categories,
or price values.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.autocomplete import generate_suggestions

router = APIRouter(tags=["Autocomplete"])


class SuggestionItem(BaseModel):
    text: str
    type: str = "phrase"
    is_correction: bool = False


class AutocompleteResponse(BaseModel):
    query: str
    suggestions: List[SuggestionItem]


@router.get("/autocomplete", response_model=AutocompleteResponse)
def autocomplete_endpoint(
    q: str = Query("", description="Partial query string for autocomplete"),
    limit: int = Query(8, ge=1, le=20, description="Maximum number of suggestions"),
    db: Session = Depends(get_db),
):
    """Generate dynamic autocomplete suggestions for a partial search query."""
    cleaned = q.strip() if q else ""
    if not cleaned:
        return AutocompleteResponse(query=q, suggestions=[])

    raw_suggestions = generate_suggestions(db=db, query=cleaned, max_results=limit)

    suggestions = [
        SuggestionItem(
            text=s.text,
            type=s.type,
            is_correction=s.is_correction,
        )
        for s in raw_suggestions
    ]

    return AutocompleteResponse(query=q, suggestions=suggestions)
