from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.search import ProductSearchResult, QueryInterpretation, SearchResponse
from app.services.search_ranking import search_products

router = APIRouter(tags=["Search"])


@router.get("/search", response_model=SearchResponse)
def search_products_endpoint(
    q: str = Query("", description="Search query string"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of search results to return"),
    min_price: Optional[float] = Query(None, ge=0, description="Optional minimum price filter override"),
    max_price: Optional[float] = Query(None, ge=0, description="Optional maximum price filter override"),
    debug: bool = Query(False, description="Enable ranking diagnostics and full ParsedQuery exposure"),
    db: Session = Depends(get_db),
):
    cleaned_query = q.strip() if q else ""
    if not cleaned_query:
        return SearchResponse(
            query=q,
            count=0,
            interpretation=QueryInterpretation(
                original_query="",
                semantic_query="",
                detected_brands=[],
                detected_categories=[],
                min_price=min_price,
                max_price=max_price,
                soft_preferences=[],
            ),
            results=[],
        )

    # Call the 4-source hybrid search ranking engine
    ranked_results, parsed = search_products(
        db=db,
        query=cleaned_query,
        limit=limit,
        min_price=min_price,
        max_price=max_price,
    )

    # Build interpretation from ParsedQuery
    interpretation_data = {
        "original_query": parsed.raw_query,
        "semantic_query": parsed.semantic_query,
        "normalized_query": parsed.normalized_query,
        "did_you_mean": parsed.did_you_mean,
        "detected_brands": parsed.detected_brands,
        "detected_categories": parsed.detected_categories,
        "min_price": min_price if min_price is not None else parsed.min_price,
        "max_price": max_price if max_price is not None else parsed.max_price,
        "soft_preferences": parsed.soft_preferences,
        "is_brand_hard_filter": parsed.is_brand_hard_filter,
        "is_category_hard_filter": parsed.is_category_hard_filter,
    }

    if debug:
        interpretation_data.update({
            "normalized_query": parsed.normalized_query,
            "detected_brand_anchor": parsed.detected_brand_anchor,
            "detected_category_anchor": parsed.detected_category_anchor,
            "is_explicit_product_query": parsed.is_explicit_product_query,
            "normalized_query_variants": parsed.normalized_query_variants,
            "tokens": [t.model_dump() for t in parsed.tokens],
        })

    interpretation = QueryInterpretation(**interpretation_data)

    results = []
    for res in ranked_results:
        p = res.product
        item_data = {
            "id": p.id,
            "product_name": p.product_name,
            "description": p.description,
            "brand": p.brand,
            "category": p.category,
            "tags": p.tags if p.tags else None,
            "price": float(p.price),
            "image": p.image,
            "final_score": round(res.final_score, 4),
        }
        if debug:
            item_data.update({
                "exact_score": round(res.exact_score, 4),
                "partial_score": round(res.partial_score, 4),
                "fuzzy_score": round(res.fuzzy_score, 4),
                "semantic_score": round(res.semantic_score, 4),
                "brand_match": res.brand_match,
                "category_match": res.category_match,
                "preference_score": round(res.preference_score, 4),
                "candidate_sources": res.candidate_sources,
            })
        results.append(ProductSearchResult(**item_data))

    return SearchResponse(
        query=q,
        count=len(results),
        interpretation=interpretation,
        results=results,
    )
