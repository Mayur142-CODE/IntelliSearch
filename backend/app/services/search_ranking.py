"""
Combined Hybrid Search Ranking Engine — Hardened v2

Architecture & Pipeline Design:
-------------------------------
1. Dynamic Query Understanding:
   - parse_query() returns ParsedQuery consumed by all downstream stages.
   - No stage re-parses the raw string.

2. 4-Source Push-Down Candidate Generation:
   - Exact: SQL match with price push-down, using SEMANTIC_QUERY (price-stripped).
   - Partial: SQL ILIKE with price push-down, using SEMANTIC_QUERY.
   - Fuzzy: pg_trgm trigram search with price push-down, using SEMANTIC_QUERY.
   - Semantic: ChromaDB vector search with price metadata push-down.

3. Candidate UNION by product ID (dedupe, keep max score per source).

4. Hard Constraints (§4.4 — defense in depth):
   - Price: mathematically exact, NEVER relaxed.
   - Brand: hard filter when explicit non-comparative query.

5. Concept Filter (§5):
   - Category anchor hard-filter for explicit product queries.

6. Hybrid Ranking (§6):
   - Fuzzy (pg_trgm) and semantic (cosine) scores are already bounded [0, 1]
     absolute similarity measures and are used AS-IS — they are NOT re-scaled
     against the current candidate set. (Fixed 2026-08: per-query min-max
     normalization previously let the "best of a weak pool" masquerade as a
     strong match, and compressed genuinely relevant candidates toward 0 in
     pools with one dominant match, causing both false positives and
     incorrectly dropped results.)
   - Named constants from search_config.py.
   - Stable tie-break: final_score DESC, semantic DESC, fuzzy DESC, product_id ASC.

7. Debug mode exposes full diagnostic breakdown per candidate.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.product import Product
from app.services.fuzzy_search import fuzzy_search_products
from app.services.query_parser import ParsedQuery, parse_query, STOP_WORDS
from app.services.semantic_search import semantic_search_products
from app.services.search_config import (
    SEMANTIC_WEIGHT, FUZZY_WEIGHT, EXACT_WEIGHT, PARTIAL_WEIGHT,
    BRAND_MATCH_BONUS, CONCEPT_MATCH_BONUS, PREFERENCE_BOOST,
    MIN_FINAL_SCORE, MIN_FINAL_SCORE_WEAK,
    STRONG_SIGNAL_FUZZY_MIN, STRONG_SIGNAL_SEMANTIC_MIN,
    CATEGORY_FUZZY_BAND,
)


@dataclass
class CombinedSearchResult:
    """Dataclass holding a matched Product instance along with individual and combined ranking scores."""
    product: Product
    exact_score: float
    partial_score: float
    fuzzy_score: float
    semantic_score: float
    final_score: float
    brand_match: bool = False
    category_match: bool = False
    preference_score: float = 0.0
    candidate_sources: List[str] = field(default_factory=list)

    def to_dict(self, debug: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.product.id,
            "product_name": self.product.product_name,
            "description": self.product.description,
            "brand": self.product.brand,
            "category": self.product.category,
            "tags": self.product.tags,
            "price": float(self.product.price),
            "image": self.product.image,
            "final_score": round(self.final_score, 4),
        }
        if debug:
            data.update({
                "exact_score": round(self.exact_score, 4),
                "partial_score": round(self.partial_score, 4),
                "fuzzy_score": round(self.fuzzy_score, 4),
                "semantic_score": round(self.semantic_score, 4),
                "brand_match": self.brand_match,
                "category_match": self.category_match,
                "preference_score": round(self.preference_score, 4),
                "candidate_sources": self.candidate_sources,
            })
        return data


# ============================================================================
# Candidate Retrieval Functions
# ============================================================================

def _get_exact_candidates(
    db: Session,
    query: str,
    limit: int = 50,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> List[Product]:
    """Fetch candidates with exact match on name, brand, category, or tags.

    Uses parameterized queries for SQL safety.
    """
    q_clean = query.strip().lower()
    if not q_clean:
        return []

    conditions = [
        or_(
            func.lower(Product.product_name) == q_clean,
            func.lower(Product.brand) == q_clean,
            func.lower(Product.category) == q_clean,
            Product.tags.ilike(f"%{q_clean}%"),
        )
    ]

    if min_price is not None:
        conditions.append(Product.price >= min_price)
    if max_price is not None:
        conditions.append(Product.price <= max_price)

    stmt = select(Product).where(*conditions).limit(limit)
    return list(db.scalars(stmt).all())


def _get_partial_candidates(
    db: Session,
    query: str,
    limit: int = 50,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> List[Product]:
    """Fetch candidates with token prefix/substring ILIKE matching."""
    q_clean = query.strip().lower()
    if not q_clean:
        return []

    tokens = [t for t in q_clean.split() if len(t) >= 2]
    if not tokens:
        return []

    match_conditions = []
    for token in tokens:
        prefix_pattern = f"{token}%"
        substring_pattern = f"%{token}%"
        match_conditions.extend([
            Product.product_name.ilike(prefix_pattern),
            Product.product_name.ilike(substring_pattern),
            Product.brand.ilike(prefix_pattern),
            Product.brand.ilike(substring_pattern),
            Product.category.ilike(prefix_pattern),
            Product.category.ilike(substring_pattern),
            Product.tags.ilike(substring_pattern),
            Product.description.ilike(substring_pattern),
        ])

    if not match_conditions:
        return []

    query_filters = [or_(*match_conditions)]
    if min_price is not None:
        query_filters.append(Product.price >= min_price)
    if max_price is not None:
        query_filters.append(Product.price <= max_price)

    stmt = select(Product).where(*query_filters).limit(limit)
    return list(db.scalars(stmt).all())


# ============================================================================
# Scoring Functions
# ============================================================================

def _calculate_exact_score(query: str, product: Product) -> float:
    """Calculate normalized exact match score (0.0 - 1.0)."""
    q_clean = query.strip().lower()
    if not q_clean:
        return 0.0

    name = (product.product_name or "").strip().lower()
    brand = (product.brand or "").strip().lower()
    category = (product.category or "").strip().lower()

    tags = [t.strip().lower() for t in product.tags.split(",") if t.strip()] if product.tags else []

    if q_clean == name:
        return 1.0
    if q_clean == brand:
        return 0.85
    if q_clean == category:
        return 0.75
    if q_clean in tags:
        return 0.70

    q_tokens = set(t for t in q_clean.split() if t)
    name_tokens = set(t for t in name.split() if t)

    if q_tokens and q_tokens == name_tokens:
        return 0.90
    if q_tokens and q_tokens.issubset(name_tokens):
        return 0.80

    return 0.0


def _calculate_partial_score(query: str, product: Product) -> float:
    """Calculate normalized prefix and partial match score (0.0 - 1.0)."""
    q_clean = query.strip().lower()
    if not q_clean:
        return 0.0

    q_tokens = [t for t in q_clean.split() if t and t not in STOP_WORDS and len(t) >= 2]
    if not q_tokens:
        q_tokens = [t for t in q_clean.split() if t]
    if not q_tokens:
        return 0.0

    name_words = [t for t in (product.product_name or "").lower().split() if t]
    brand_words = [t for t in (product.brand or "").lower().split() if t]
    cat_words = [t for t in (product.category or "").lower().split() if t]

    tag_words = [t.strip().lower() for t in product.tags.split(",") if t.strip()] if product.tags else []

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
                    prefix_ratio = len(qt) / float(len(tw))
                    token_max = max(token_max, 0.85 * max(0.70, prefix_ratio))
                elif qt in tw and len(qt) >= 3:
                    sub_ratio = len(qt) / float(len(tw))
                    token_max = max(token_max, 0.50 * max(0.60, sub_ratio))
            scores.append(token_max)
        return sum(scores) / float(len(q_tokens)) if scores else 0.0

    name_score = _token_field_score(name_words)
    brand_score = _token_field_score(brand_words)
    cat_score = _token_field_score(cat_words)
    tags_score = _token_field_score(tag_words)

    partial_combined = (
        (name_score * 0.50) +
        (brand_score * 0.25) +
        (cat_score * 0.15) +
        (tags_score * 0.10)
    )

    return min(max(partial_combined, 0.0), 1.0)


def _calculate_preference_score(
    parsed: ParsedQuery,
    product: Product,
    min_cand_price: float,
    max_cand_price: float,
) -> float:
    """Calculate soft preference ranking boost based on relative candidate context."""
    if not parsed.soft_preferences:
        return 0.0

    pref_score = 0.0
    price_val = float(product.price)
    price_range = max(1.0, max_cand_price - min_cand_price)

    # Budget / Cheap / Affordable: relative lower price boost
    if any(p in parsed.soft_preferences for p in ("budget", "affordable", "cheap")):
        rel_pos = 1.0 - ((price_val - min_cand_price) / price_range)
        pref_score = max(pref_score, rel_pos * 0.8)

    # Premium / Luxury: relative higher price boost
    if any(p in parsed.soft_preferences for p in ("premium", "luxury", "high-end")):
        rel_pos = (price_val - min_cand_price) / price_range
        pref_score = max(pref_score, rel_pos * 0.8)

    # Textual attribute preferences (e.g. lightweight, waterproof, comfortable)
    product_text = f"{product.product_name} {product.description} {product.tags}".lower()
    for pref in parsed.soft_preferences:
        if pref not in ("budget", "affordable", "cheap", "premium", "luxury", "high-end"):
            if pref in product_text:
                pref_score = max(pref_score, 0.5)

    return min(max(pref_score, 0.0), 1.0)


# ============================================================================
# §5 — Concept Filter
# ============================================================================

def _apply_concept_filter(
    candidates: Dict[int, Dict[str, Any]],
    parsed: ParsedQuery,
) -> Dict[int, Dict[str, Any]]:
    """Apply concept filter: when a category anchor is detected, prefer
    products in that category (or closely related categories).

    This is NOT a hard filter that removes all non-matching products —
    rather, it removes candidates whose category is clearly unrelated
    (accessories vs core products) when the query is explicitly about
    a product type.
    """
    if not parsed.detected_category_anchor or not parsed.is_explicit_product_query:
        return candidates

    anchor = parsed.detected_category_anchor.lower()

    # Identify which candidates match the anchor category
    matching_ids = set()
    non_matching_ids = set()

    for p_id, data in candidates.items():
        product_cat = (data["product"].category or "").lower()
        # Check exact match or fuzzy match within band
        if product_cat == anchor:
            matching_ids.add(p_id)
        else:
            cat_similarity = fuzz.ratio(product_cat, anchor)
            if cat_similarity >= CATEGORY_FUZZY_BAND:
                matching_ids.add(p_id)
            else:
                non_matching_ids.add(p_id)

    # If we have matching products, only keep them
    # If no matching products at all, return all (don't zero-out results)
    if matching_ids:
        return {pid: candidates[pid] for pid in matching_ids}

    return candidates


# ============================================================================
# Main Search Function
# ============================================================================

def search_products(
    db: Session,
    query: str,
    limit: int = 20,
    candidate_limit: int = 50,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_final_score: float = MIN_FINAL_SCORE,
    parsed_query: Optional[ParsedQuery] = None,
) -> Tuple[List[CombinedSearchResult], ParsedQuery]:
    """Perform 4-source hybrid search with dynamic query understanding,
    push-down filtering, candidate union, hard constraint verification,
    concept filtering, and multi-signal reranking.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        empty_parsed = ParsedQuery(raw_query="", semantic_query="")
        return [], empty_parsed

    # ===================================================================
    # 1. Dynamic Query Understanding
    # ===================================================================
    parsed = parsed_query or parse_query(db, cleaned_query)

    # Resolve effective price boundaries (explicit params take priority)
    eff_min_price = min_price if min_price is not None else parsed.min_price
    eff_max_price = max_price if max_price is not None else parsed.max_price

    # The SEARCH QUERY for all retrieval paths is the semantic_query
    # (price-stripped, operator-cleaned) — NOT the raw query
    search_text = parsed.semantic_query if parsed.semantic_query else cleaned_query

    # ===================================================================
    # 2. 4-Source Push-Down Candidate Retrieval
    # ===================================================================
    # Exact Candidates (PostgreSQL)
    exact_candidates = _get_exact_candidates(
        db, search_text, limit=candidate_limit,
        min_price=eff_min_price, max_price=eff_max_price,
    )

    # Partial/Prefix Candidates (PostgreSQL)
    partial_candidates = _get_partial_candidates(
        db, search_text, limit=candidate_limit,
        min_price=eff_min_price, max_price=eff_max_price,
    )

    # Fuzzy Candidates (pg_trgm) — use semantic query
    fuzzy_candidates = fuzzy_search_products(
        db=db,
        query=search_text,
        limit=candidate_limit,
        min_price=eff_min_price,
        max_price=eff_max_price,
        min_similarity=0.01,
    )

    # Semantic Candidates (ChromaDB) — use normalized variants
    semantic_candidates = semantic_search_products(
        db=db,
        query=parsed.semantic_query if parsed.semantic_query else search_text,
        limit=candidate_limit,
        min_price=eff_min_price,
        max_price=eff_max_price,
        min_similarity=0.38,
        normalized_queries=parsed.normalized_query_variants,
    )

    # ===================================================================
    # 3. Candidate UNION by Product ID
    # ===================================================================
    candidate_map: Dict[int, Dict[str, Any]] = {}

    for p in exact_candidates:
        candidate_map[p.id] = {
            "product": p,
            "fuzzy_score": 0.0,
            "semantic_score": 0.0,
            "sources": {"exact"},
        }

    for p in partial_candidates:
        if p.id not in candidate_map:
            candidate_map[p.id] = {
                "product": p,
                "fuzzy_score": 0.0,
                "semantic_score": 0.0,
                "sources": {"partial"},
            }
        else:
            candidate_map[p.id]["sources"].add("partial")

    for item in fuzzy_candidates:
        p_id = item.product.id
        if p_id in candidate_map:
            candidate_map[p_id]["fuzzy_score"] = max(0.0, float(item.fuzzy_score))
            candidate_map[p_id]["sources"].add("fuzzy")
        else:
            candidate_map[p_id] = {
                "product": item.product,
                "fuzzy_score": max(0.0, float(item.fuzzy_score)),
                "semantic_score": 0.0,
                "sources": {"fuzzy"},
            }

    for item in semantic_candidates:
        p_id = item.product.id
        if p_id in candidate_map:
            candidate_map[p_id]["semantic_score"] = max(0.0, float(item.semantic_score))
            candidate_map[p_id]["sources"].add("semantic")
        else:
            candidate_map[p_id] = {
                "product": item.product,
                "fuzzy_score": 0.0,
                "semantic_score": max(0.0, float(item.semantic_score)),
                "sources": {"semantic"},
            }

    if not candidate_map:
        return [], parsed

    # ===================================================================
    # 4. Absolute Hard-Filter Safety Verification (§4.4)
    # ===================================================================
    verified_candidates: Dict[int, Dict[str, Any]] = {}

    for p_id, data in candidate_map.items():
        p: Product = data["product"]
        p_price = float(p.price)

        # 4a. Absolute Price Check — NEVER relaxed, defense in depth
        if eff_min_price is not None and p_price < eff_min_price:
            continue
        if eff_max_price is not None and p_price > eff_max_price:
            continue

        # 4b. Explicit Hard Brand Filter
        if parsed.is_brand_hard_filter and parsed.detected_brands:
            brand_matched = any(
                p.brand.strip().lower() == b.strip().lower()
                for b in parsed.detected_brands
            )
            if not brand_matched:
                continue

        verified_candidates[p_id] = data

    if not verified_candidates:
        return [], parsed

    # ===================================================================
    # 5. Concept Filter (§5)
    # ===================================================================
    verified_candidates = _apply_concept_filter(verified_candidates, parsed)

    if not verified_candidates:
        return [], parsed

    # Price range for relative soft preference scoring
    all_prices = [float(d["product"].price) for d in verified_candidates.values()]
    min_cand_price = min(all_prices) if all_prices else 0.0
    max_cand_price = max(all_prices) if all_prices else 1000.0

    # ===================================================================
    # 6. Multi-Signal Scoring & Reranking (§6)
    # ===================================================================

    # NOTE: fuzzy_score (pg_trgm similarity) and semantic_score (cosine similarity)
    # are ALREADY absolute, bounded [0, 1] similarity measures. We intentionally do
    # NOT min-max normalize them against the current candidate set — doing so was
    # the root cause of bad ranking: a per-query pool of only weak matches would
    # stretch its best (still-weak) candidate up to 1.0, letting irrelevant
    # products masquerade as strong matches, while genuinely relevant candidates
    # in a pool with one dominant match would get compressed toward 0 and
    # incorrectly dropped by the MIN_FINAL_SCORE threshold. Raw scores keep the
    # STRONG_SIGNAL_* gating and the weighted sum meaningful in absolute terms.
    ranked_results: List[CombinedSearchResult] = []

    for p_id, data in verified_candidates.items():
        product: Product = data["product"]
        f_score_raw = max(data["fuzzy_score"], 0.0)
        s_score_raw = max(data["semantic_score"], 0.0)

        # Calculate exact and partial scores using the search text
        e_score = _calculate_exact_score(search_text, product)
        p_score = _calculate_partial_score(search_text, product)

        # Brand & Category Matches
        brand_match = (
            any(b.lower() == (product.brand or "").lower() for b in parsed.detected_brands)
            if parsed.detected_brands
            else False
        )
        category_match = (
            (product.category or "").lower() == (parsed.detected_category_anchor or "").lower()
            if parsed.detected_category_anchor
            else False
        )

        # Soft preference score
        pref_score = _calculate_preference_score(
            parsed, product, min_cand_price, max_cand_price
        )

        # Base weighted score (§6)
        final_score = (
            (e_score * EXACT_WEIGHT) +
            (p_score * PARTIAL_WEIGHT) +
            (f_score_raw * FUZZY_WEIGHT) +
            (s_score_raw * SEMANTIC_WEIGHT)
        )

        # Add bonuses
        if brand_match:
            final_score += BRAND_MATCH_BONUS
        if category_match:
            final_score += CONCEPT_MATCH_BONUS
        if pref_score > 0:
            final_score += pref_score * PREFERENCE_BOOST

        final_score = min(max(final_score, 0.0), 1.0)

        # Generic False-Positive Protection
        has_strong_signal = (
            (e_score > 0) or (p_score >= 0.25) or
            (f_score_raw >= STRONG_SIGNAL_FUZZY_MIN) or
            (s_score_raw >= STRONG_SIGNAL_SEMANTIC_MIN) or
            brand_match
        )
        effective_threshold = min_final_score if has_strong_signal else MIN_FINAL_SCORE_WEAK

        if final_score < effective_threshold:
            continue

        ranked_results.append(
            CombinedSearchResult(
                product=product,
                exact_score=e_score,
                partial_score=p_score,
                fuzzy_score=f_score_raw,
                semantic_score=s_score_raw,
                final_score=final_score,
                brand_match=brand_match,
                category_match=category_match,
                preference_score=pref_score,
                candidate_sources=sorted(list(data["sources"])),
            )
        )

    # ===================================================================
    # 7. Deterministic Sort — stable tie-break (§6)
    # ===================================================================
    ranked_results.sort(
        key=lambda r: (
            r.final_score,
            r.semantic_score,
            r.fuzzy_score,
            -r.product.id,  # stable tie-break by product_id
        ),
        reverse=True,
    )

    return ranked_results[:limit], parsed
