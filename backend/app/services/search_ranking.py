"""
Combined Search Ranking Engine

Architecture Design & 4-Source Candidate Generation:
---------------------------------------------------
1. Candidate Generation Layer (4 Independent Sources):
   - Exact Candidates: Direct SQL query for exact matches on product_name, brand, category, or tags.
   - Partial Candidates: Direct SQL ILIKE query for token prefix and substring matches.
   - Fuzzy Candidates: PostgreSQL pg_trgm trigram fuzzy search candidates (top N).
   - Semantic Candidates: FastEmbed vector cosine similarity candidates (top N).

2. Candidate Union & Multi-Signal Scoring Engine (Scores 0.0 - 1.0):
   - Forms a complete UNION of candidates by product ID across all 4 sources.
   - Evaluates:
     * exact_score (20% weight): Exact string & token set matches.
     * partial_score (15% weight): Prefix and token substring matches across attributes.
     * fuzzy_score (30% weight): PostgreSQL pg_trgm similarity score.
     * semantic_score (35% weight): FastEmbed dense vector cosine similarity score.

3. Final Weighted Ranking Formula (Unchanged):
   final_score = (exact_score * 0.20) + (partial_score * 0.15) + (fuzzy_score * 0.30) + (semantic_score * 0.35)
"""

from dataclasses import dataclass
from typing import Any, Dict, List
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.product import Product
from app.services.fuzzy_search import fuzzy_search_products
from app.services.semantic_search import semantic_search_products

# Scoring Signal Weights (Tunable Constants)
EXACT_WEIGHT = 0.20
PARTIAL_WEIGHT = 0.15
FUZZY_WEIGHT = 0.30
SEMANTIC_WEIGHT = 0.35


@dataclass
class CombinedSearchResult:
    """Dataclass holding a matched Product instance along with individual and combined ranking scores."""
    product: Product
    exact_score: float
    partial_score: float
    fuzzy_score: float
    semantic_score: float
    final_score: float

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
            "exact_score": round(self.exact_score, 4),
            "partial_score": round(self.partial_score, 4),
            "fuzzy_score": round(self.fuzzy_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "final_score": round(self.final_score, 4),
        }


def _get_exact_candidates(db: Session, query: str, limit: int = 50) -> List[Product]:
    """Fetch candidates from PostgreSQL with exact/case-insensitive match on name, brand, category, or tags."""
    q_clean = query.strip().lower()
    if not q_clean:
        return []

    stmt = (
        select(Product)
        .where(
            or_(
                func.lower(Product.product_name) == q_clean,
                func.lower(Product.brand) == q_clean,
                func.lower(Product.category) == q_clean,
                Product.tags.ilike(f"%{q_clean}%"),
            )
        )
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


def _get_partial_candidates(db: Session, query: str, limit: int = 50) -> List[Product]:
    """Fetch candidates from PostgreSQL with token prefix and substring ILIKE matching."""
    q_clean = query.strip().lower()
    if not q_clean:
        return []

    tokens = [t for t in q_clean.split() if t]
    if not tokens:
        return []

    conditions = []
    for token in tokens:
        if len(token) < 2:
            continue
        prefix_pattern = f"{token}%"
        substring_pattern = f"%{token}%"
        conditions.extend([
            Product.product_name.ilike(prefix_pattern),
            Product.product_name.ilike(substring_pattern),
            Product.brand.ilike(prefix_pattern),
            Product.category.ilike(prefix_pattern),
            Product.tags.ilike(substring_pattern),
        ])

    if not conditions:
        return []

    stmt = (
        select(Product)
        .where(or_(*conditions))
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


def _calculate_exact_score(query: str, product: Product) -> float:
    """
    Calculate normalized exact match score (0.0 - 1.0).
    Strongest weight for full product_name match, followed by brand, category, tags, and exact token sets.
    """
    q_clean = query.strip().lower()
    if not q_clean:
        return 0.0

    name = (product.product_name or "").strip().lower()
    brand = (product.brand or "").strip().lower()
    category = (product.category or "").strip().lower()
    
    tags = []
    if product.tags:
        if isinstance(product.tags, list):
            tags = [t.strip().lower() for t in product.tags]
        else:
            tags = [t.strip().lower() for t in str(product.tags).split(",") if t.strip()]

    # 1. Product Name Exact Full Match
    if q_clean == name:
        return 1.0

    # 2. Brand Exact Full Match
    if q_clean == brand:
        return 0.85

    # 3. Category Exact Full Match
    if q_clean == category:
        return 0.75

    # 4. Tag Exact Match
    if q_clean in tags:
        return 0.70

    # 5. Token Set Exact Match (all query tokens match product_name tokens)
    q_tokens = set(t for t in q_clean.split() if t)
    name_tokens = set(t for t in name.split() if t)

    if q_tokens and q_tokens == name_tokens:
        return 0.90

    if q_tokens and q_tokens.issubset(name_tokens):
        return 0.80

    return 0.0


def _calculate_partial_score(query: str, product: Product) -> float:
    """
    Calculate normalized prefix and partial match score (0.0 - 1.0).
    Evaluates prefix and token overlap across product_name (50%), brand (25%), category (15%), and tags (10%).
    """
    q_clean = query.strip().lower()
    if not q_clean:
        return 0.0

    q_tokens = [t for t in q_clean.split() if t]
    if not q_tokens:
        return 0.0

    name_words = [t for t in (product.product_name or "").lower().split() if t]
    brand_words = [t for t in (product.brand or "").lower().split() if t]
    cat_words = [t for t in (product.category or "").lower().split() if t]

    tag_words = []
    if product.tags:
        if isinstance(product.tags, list):
            tag_words = [t.strip().lower() for t in product.tags]
        else:
            tag_words = [t.strip().lower() for t in str(product.tags).split(",") if t.strip()]

    def _token_field_score(target_words: List[str]) -> float:
        if not target_words:
            return 0.0

        scores = []
        for qt in q_tokens:
            token_max = 0.0
            for tw in target_words:
                if tw == qt:
                    token_max = max(token_max, 1.0)
                elif tw.startswith(qt) and len(qt) >= 2:
                    # Prefix match score weighted by length ratio with baseline
                    prefix_ratio = len(qt) / float(len(tw))
                    token_max = max(token_max, 0.85 * max(0.70, prefix_ratio))
                elif qt in tw and len(qt) >= 3:
                    # Substring match score
                    sub_ratio = len(qt) / float(len(tw))
                    token_max = max(token_max, 0.50 * max(0.60, sub_ratio))
            scores.append(token_max)

        return sum(scores) / float(len(q_tokens)) if scores else 0.0

    name_score = _token_field_score(name_words)
    brand_score = _token_field_score(brand_words)
    cat_score = _token_field_score(cat_words)
    tags_score = _token_field_score(tag_words)

    # Weighted combination of partial match signals across product attributes
    partial_combined = (
        (name_score * 0.50) +
        (brand_score * 0.25) +
        (cat_score * 0.15) +
        (tags_score * 0.10)
    )

    return min(max(partial_combined, 0.0), 1.0)


def search_products(
    db: Session,
    query: str,
    limit: int = 20,
    candidate_limit: int = 50,
    min_final_score: float = 0.15,
) -> List[CombinedSearchResult]:
    """
    Perform hybrid product search with a 4-way candidate generation layer.

    Candidate Sources:
    1. Exact Candidates (PostgreSQL query)
    2. Prefix/Partial Candidates (PostgreSQL ILIKE query)
    3. Fuzzy Candidates (PostgreSQL pg_trgm trigram search)
    4. Semantic Candidates (FastEmbed dense vector cosine similarity)

    Union of all 4 candidate sets is evaluated against exact, partial, fuzzy, and semantic scoring functions.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    # 1. Exact Candidates from PostgreSQL
    exact_candidates = _get_exact_candidates(db, cleaned_query, limit=candidate_limit)

    # 2. Prefix/Partial Candidates from PostgreSQL
    partial_candidates = _get_partial_candidates(db, cleaned_query, limit=candidate_limit)

    # 3. Fuzzy Candidates from pg_trgm
    fuzzy_candidates = fuzzy_search_products(
        db=db,
        query=cleaned_query,
        limit=candidate_limit,
        min_similarity=0.01,
    )

    # 4. Semantic Candidates from FastEmbed
    semantic_candidates = semantic_search_products(
        db=db,
        query=cleaned_query,
        limit=candidate_limit,
        min_similarity=0.0,
    )

    # 5. Form Candidate UNION Map by Product ID
    candidate_map: Dict[int, Dict[str, Any]] = {}

    for p in exact_candidates:
        candidate_map[p.id] = {
            "product": p,
            "fuzzy_score": 0.0,
            "semantic_score": 0.0,
        }

    for p in partial_candidates:
        if p.id not in candidate_map:
            candidate_map[p.id] = {
                "product": p,
                "fuzzy_score": 0.0,
                "semantic_score": 0.0,
            }

    for item in fuzzy_candidates:
        p_id = item.product.id
        if p_id in candidate_map:
            candidate_map[p_id]["fuzzy_score"] = max(0.0, float(item.fuzzy_score))
        else:
            candidate_map[p_id] = {
                "product": item.product,
                "fuzzy_score": max(0.0, float(item.fuzzy_score)),
                "semantic_score": 0.0,
            }

    for item in semantic_candidates:
        p_id = item.product.id
        if p_id in candidate_map:
            candidate_map[p_id]["semantic_score"] = max(0.0, float(item.semantic_score))
        else:
            candidate_map[p_id] = {
                "product": item.product,
                "fuzzy_score": 0.0,
                "semantic_score": max(0.0, float(item.semantic_score)),
            }

    if not candidate_map:
        return []

    # 6. Calculate multi-signal scores for each candidate
    ranked_results: List[CombinedSearchResult] = []

    for p_id, data in candidate_map.items():
        product: Product = data["product"]
        f_score: float = min(max(data["fuzzy_score"], 0.0), 1.0)
        s_score: float = min(max(data["semantic_score"], 0.0), 1.0)

        e_score: float = _calculate_exact_score(cleaned_query, product)
        p_score: float = _calculate_partial_score(cleaned_query, product)

        # Final weighted score combination (unchanged formula)
        final_score = (
            (e_score * EXACT_WEIGHT) +
            (p_score * PARTIAL_WEIGHT) +
            (f_score * FUZZY_WEIGHT) +
            (s_score * SEMANTIC_WEIGHT)
        )

        if final_score < min_final_score:
            continue

        ranked_results.append(
            CombinedSearchResult(
                product=product,
                exact_score=e_score,
                partial_score=p_score,
                fuzzy_score=f_score,
                semantic_score=s_score,
                final_score=final_score,
            )
        )

    # 7. Deterministic sorting by final_score DESC with tie-breakers
    ranked_results.sort(
        key=lambda r: (
            r.final_score,
            r.semantic_score,
            r.fuzzy_score,
            -r.product.id,
        ),
        reverse=True,
    )

    return ranked_results[:limit]
