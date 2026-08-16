from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sqlalchemy import case, desc, func, select
from sqlalchemy.orm import Session

from app.models.product import Product


@dataclass
class FuzzySearchResult:
    """Dataclass holding a matched Product instance along with its PostgreSQL fuzzy score."""
    product: Product
    fuzzy_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.product.id,
            "product_name": self.product.product_name,
            "description": self.product.description,
            "brand": self.product.brand,
            "category": self.product.category,
            "tags": self.product.tags,
            "price": float(self.product.price),
            "image": self.product.image,
            "fuzzy_score": round(self.fuzzy_score, 4),
        }


def _build_token_score_expr(token: str):
    """
    Build field-weighted trigram similarity expression for a single query token.
    Field Weights: Product Name (50%), Brand (20%), Category (15%), Tags (15%).
    """
    name_sim = func.greatest(
        func.word_similarity(token, Product.product_name),
        func.similarity(token, Product.product_name),
    )
    brand_sim = func.greatest(
        func.word_similarity(token, Product.brand),
        func.similarity(token, Product.brand),
    )
    cat_sim = func.greatest(
        func.word_similarity(token, Product.category),
        func.similarity(token, Product.category),
    )
    tags_sim = func.word_similarity(token, Product.tags)

    return (
        (name_sim * 0.50) +
        (brand_sim * 0.20) +
        (cat_sim * 0.15) +
        (tags_sim * 0.15)
    )


def fuzzy_search_products(
    db: Session,
    query: str,
    limit: int = 20,
    min_similarity: float = 0.15,
) -> List[FuzzySearchResult]:
    """
    Perform token-aware & field-weighted PostgreSQL pg_trgm fuzzy search across products.

    Pushes token decomposition and multi-token coverage scoring entirely into PostgreSQL.
    Products matching multiple search terms receive significant score boosts over single-term matches.

    Args:
        db: SQLAlchemy database session.
        query: Raw search query string.
        limit: Maximum number of candidate results to return (default 20).
        min_similarity: Minimum fuzzy similarity score threshold (default 0.15).

    Returns:
        List of FuzzySearchResult objects containing matched Product models and fuzzy scores.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    # Tokenize query into individual non-empty terms
    tokens = [t.lower() for t in cleaned_query.split() if t.strip()]
    if not tokens:
        return []

    num_tokens = len(tokens)

    if num_tokens == 1:
        single_token = tokens[0]
        token_score = _build_token_score_expr(single_token)
        full_name_sim = func.greatest(
            func.word_similarity(cleaned_query, Product.product_name),
            func.similarity(cleaned_query, Product.product_name),
        )
        fuzzy_score_expr = (token_score * 0.75 + full_name_sim * 0.25).label("fuzzy_score")
    else:
        # Evaluate individual token scores across product fields
        token_exprs = [_build_token_score_expr(t) for t in tokens]

        # Calculate average token score
        sum_token_expr = token_exprs[0]
        for expr in token_exprs[1:]:
            sum_token_expr = sum_token_expr + expr
        avg_token_expr = sum_token_expr / float(num_tokens)

        # Calculate token coverage (how many tokens scored >= 0.15 threshold)
        match_cases = [case((expr >= 0.15, 1.0), else_=0.0) for expr in token_exprs]
        sum_matches = match_cases[0]
        for match_case in match_cases[1:]:
            sum_matches = sum_matches + match_case
        coverage_ratio_expr = sum_matches / float(num_tokens)

        # Full query similarity as a secondary signal
        full_name_sim = func.greatest(
            func.word_similarity(cleaned_query, Product.product_name),
            func.similarity(cleaned_query, Product.product_name),
        )

        # Multi-token scoring formula:
        # Base Average Token Score (60%) + Token Coverage Bonus (25%) + Full Query Match (15%)
        fuzzy_score_expr = (
            (avg_token_expr * 0.60) +
            (avg_token_expr * coverage_ratio_expr * 0.25) +
            (full_name_sim * 0.15)
        ).label("fuzzy_score")

    # Construct SQL query executed entirely inside PostgreSQL
    stmt = (
        select(Product, fuzzy_score_expr)
        .where(fuzzy_score_expr >= min_similarity)
        .order_by(desc(fuzzy_score_expr))
        .limit(limit)
    )

    results = db.execute(stmt).all()

    return [
        FuzzySearchResult(
            product=row.Product,
            fuzzy_score=float(row.fuzzy_score),
        )
        for row in results
    ]
