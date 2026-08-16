from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.search import ProductSearchResult, SearchResponse
from app.services.search_ranking import search_products

router = APIRouter(tags=["Search"])


@router.get("/search", response_model=SearchResponse)
def search_products_endpoint(
    q: str = Query("", description="Search query string"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of search results to return"),
    db: Session = Depends(get_db),
):
    cleaned_query = q.strip() if q else ""
    if not cleaned_query:
        return SearchResponse(
            query=q,
            count=0,
            results=[],
        )

    # Call the 4-source hybrid search ranking engine
    ranked_results = search_products(
        db=db,
        query=cleaned_query,
        limit=limit,
    )

    results = [
        ProductSearchResult(
            id=res.product.id,
            product_name=res.product.product_name,
            brand=res.product.brand,
            category=res.product.category,
            price=float(res.product.price),
            image=res.product.image,
            exact_score=round(res.exact_score, 4),
            partial_score=round(res.partial_score, 4),
            fuzzy_score=round(res.fuzzy_score, 4),
            semantic_score=round(res.semantic_score, 4),
            final_score=round(res.final_score, 4),
        )
        for res in ranked_results
    ]

    return SearchResponse(
        query=q,
        count=len(results),
        results=results,
    )
