from typing import List, Optional
from pydantic import BaseModel, Field


class ProductSearchResult(BaseModel):
    id: int
    product_name: str
    brand: str
    category: str
    price: float
    image: Optional[str] = None

    exact_score: float = Field(..., description="Exact string/token match score (0.0 - 1.0)")
    partial_score: float = Field(..., description="Prefix/partial token match score (0.0 - 1.0)")
    fuzzy_score: float = Field(..., description="PostgreSQL pg_trgm trigram similarity score (0.0 - 1.0)")
    semantic_score: float = Field(..., description="FastEmbed dense vector cosine similarity score (0.0 - 1.0)")
    final_score: float = Field(..., description="Combined weighted relevance score (0.0 - 1.0)")

    model_config = {"from_attributes": True}


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[ProductSearchResult]
