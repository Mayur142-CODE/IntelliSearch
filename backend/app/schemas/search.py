from typing import List, Optional
from pydantic import BaseModel, Field


class ProductSearchResult(BaseModel):
    id: int
    product_name: str
    description: Optional[str] = None
    brand: str
    category: str
    tags: Optional[str] = None
    price: float
    image: Optional[str] = None
    final_score: float = Field(..., description="Combined weighted relevance score (0.0 - 1.0)")

    # Diagnostics (exposed when debug=True)
    exact_score: Optional[float] = Field(None, description="Exact string/token match score (0.0 - 1.0)")
    partial_score: Optional[float] = Field(None, description="Prefix/partial token match score (0.0 - 1.0)")
    fuzzy_score: Optional[float] = Field(None, description="PostgreSQL pg_trgm trigram similarity score (0.0 - 1.0)")
    semantic_score: Optional[float] = Field(None, description="ChromaDB dense vector cosine similarity score (0.0 - 1.0)")
    brand_match: Optional[bool] = Field(None, description="Whether brand matched parsed query intent")
    category_match: Optional[bool] = Field(None, description="Whether category matched parsed query intent")
    preference_score: Optional[float] = Field(None, description="Soft preference relative boost score")
    candidate_sources: Optional[List[str]] = Field(None, description="Candidate retrieval sources that discovered this item")

    model_config = {"from_attributes": True}


class QueryInterpretation(BaseModel):
    original_query: str
    semantic_query: str
    normalized_query: Optional[str] = None
    detected_brands: List[str] = []
    detected_categories: List[str] = []
    detected_brand_anchor: Optional[str] = None
    detected_category_anchor: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    is_explicit_product_query: bool = False
    is_brand_hard_filter: bool = False
    is_category_hard_filter: bool = False
    soft_preferences: List[str] = []
    normalized_query_variants: Optional[List[str]] = None
    tokens: Optional[List[dict]] = None


class SearchResponse(BaseModel):
    query: str
    count: int
    interpretation: Optional[QueryInterpretation] = None
    results: List[ProductSearchResult]
