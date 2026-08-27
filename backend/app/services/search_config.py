"""
Centralized Search Configuration — All Tunable Constants

This module contains every configurable weight, threshold, and scoring
parameter used across the search pipeline. No bare numbers should appear
inline in query_parser, search_ranking, fuzzy_search, or semantic_search.

GROUND RULE: These are generic constants that apply to every query.
Nothing here is keyed by a literal query string, brand name, or product name.
"""

# ============================================================================
# §3.3 — Typo Correction Confidence Formula Weights
# ============================================================================
# confidence = W_SIM * normalized_similarity
#            + W_FREQ * log1p(catalog_frequency) / log1p(max_catalog_frequency)
#            + W_FIELD_PRIORITY * field_priority_bonus

W_SIM: float = 0.80           # Weight for normalized RapidFuzz/Levenshtein similarity score
W_FREQ: float = 0.10          # Weight for log-normalized catalog frequency
W_FIELD_PRIORITY: float = 0.10  # Weight for field-type priority bonus (tie-breaker)

# Field priority order (higher = more trusted for correction)
FIELD_PRIORITY_BONUS = {
    "brand": 1.0,
    "category": 0.90,
    "product_name": 0.80,
    "tag": 0.70,
    "description": 0.50,
}

# ============================================================================
# §3.4 — Length-Gated Correction Thresholds
# ============================================================================
# Protects short tokens like HP, TV, PC, 3M from mangling.
# token length ≤ 2: no correction unless exact catalog match
# token length 3–4: min confidence ≥ 0.75
# token length ≥ 5: min confidence ≥ 0.70

MIN_CONFIDENCE_SHORT: float = 0.75   # For tokens 3-4 chars
MIN_CONFIDENCE_LONG: float = 0.70    # For tokens ≥ 5 chars

# ============================================================================
# §3.5 — Multi-Candidate Margin
# ============================================================================
# If top two candidates are within this margin of each other in confidence,
# retain both as OR'd candidates for retrieval.

MULTI_CANDIDATE_MARGIN: float = 0.03

# ============================================================================
# §3.2 — Typo Correction Distance Thresholds
# ============================================================================
# Maximum Levenshtein distance allowed for candidate generation

MAX_EDIT_DISTANCE_SHORT: int = 1    # For tokens 3-4 chars
MAX_EDIT_DISTANCE_LONG: int = 2     # For tokens ≥ 5 chars
MAX_EDIT_DISTANCE_PHONETIC: int = 3  # Expanded threshold when phonetic match is strong

# ============================================================================
# §6 — Hybrid Ranking Signal Weights
# ============================================================================
# Combined score formula:
# final_score = EXACT_WEIGHT * exact_score
#             + PARTIAL_WEIGHT * partial_score
#             + FUZZY_WEIGHT * fuzzy_score
#             + SEMANTIC_WEIGHT * semantic_score
#             + BRAND_MATCH_BONUS (if brand anchor matches)
#             + CONCEPT_MATCH_BONUS (if category anchor matches)

EXACT_WEIGHT: float = 0.20      # Weight for exact string/token match signal
PARTIAL_WEIGHT: float = 0.15    # Weight for prefix/partial match signal
FUZZY_WEIGHT: float = 0.25      # Weight for PostgreSQL pg_trgm fuzzy signal
SEMANTIC_WEIGHT: float = 0.35   # Weight for ChromaDB vector similarity signal

BRAND_MATCH_BONUS: float = 0.05    # Additive bonus when detected brand matches product brand
CONCEPT_MATCH_BONUS: float = 0.04  # Additive bonus when detected category matches product category
PREFERENCE_BOOST: float = 0.05     # Scaling factor for soft preference score contribution

# ============================================================================
# §6 — Scoring Thresholds
# ============================================================================

MIN_FINAL_SCORE: float = 0.10        # Minimum final score for inclusion (with strong signal)
MIN_FINAL_SCORE_WEAK: float = 0.15   # Minimum final score when no strong signal present

# Strong signal thresholds (at least one must be met)
STRONG_SIGNAL_FUZZY_MIN: float = 0.30
STRONG_SIGNAL_SEMANTIC_MIN: float = 0.40

# ============================================================================
# §5 — Concept Filter Thresholds
# ============================================================================

CATEGORY_MATCH_CONFIDENCE_MIN: float = 85.0  # Minimum confidence to treat as category anchor
CATEGORY_FUZZY_BAND: float = 80.0            # Fuzzy similarity band for related categories

# ============================================================================
# Price Parser Constants
# ============================================================================

# No hardcoded price values — these are structural parameters only
PRICE_SWAP_ON_INVALID_RANGE: bool = True  # If min > max, swap (vs. reject as invalid)
