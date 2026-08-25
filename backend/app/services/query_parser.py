"""
Dynamic Query Understanding & Parsing Module — Hardened v2

Architecture & Design Principles:
---------------------------------
1. Generic NLP Parsing (Zero hardcoded brands, categories, or typo dictionaries).
2. Dynamic Vocabulary: Discovered at runtime from PostgreSQL, with SEPARATE
   per-field indexes (brand, category, product_name, tag) so field-type priority
   can break ties without one large pool drowning the other.
3. Dynamic Typo Correction:
   - Confidence formula: W_SIM * similarity + W_FREQ * freq_score + W_FIELD_PRIORITY * field_bonus
   - Length-gated thresholds: ≤2 chars no correction (unless exact), 3-4 chars ≥0.88, ≥5 chars ≥0.80
   - Scorer selection: fuzz.ratio for single-token, token_sort_ratio for multi-word
   - Phonetic matching ungated from Levenshtein for candidate generation
   - Multi-candidate OR when top candidates within margin
4. Price Parser:
   - Supports k/K/thousand suffixes, currency symbols (₹, $, Rs., rs)
   - RANGE parsed before UNDER/ABOVE to avoid misread
   - Price spans removed from ALL downstream text (semantic, fuzzy, exact, partial)
5. Full pipeline returns ParsedQuery Pydantic model consumed by all downstream stages.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, distance
from sqlalchemy import distinct, select
from sqlalchemy.orm import Session

from app.models.product import Product
from app.services.search_config import (
    W_SIM, W_FREQ, W_FIELD_PRIORITY,
    FIELD_PRIORITY_BONUS,
    MIN_CONFIDENCE_SHORT, MIN_CONFIDENCE_LONG,
    MULTI_CANDIDATE_MARGIN,
    MAX_EDIT_DISTANCE_SHORT, MAX_EDIT_DISTANCE_LONG, MAX_EDIT_DISTANCE_PHONETIC,
    CATEGORY_MATCH_CONFIDENCE_MIN,
    PRICE_SWAP_ON_INVALID_RANGE,
)


# ============================================================================
# Common English stop words — prevents false-positive matching
# ============================================================================
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "to", "for", "with", "in", "on", "at", "by",
    "from", "of", "up", "than", "less", "more", "my", "is", "it", "as", "be",
    "this", "that", "some", "any", "all", "get", "good", "best", "new", "i",
    "me", "we", "you", "he", "she", "they", "can", "not", "do", "did", "has",
    "have", "had", "will", "would", "should", "could", "may", "might",
}

# Comparison indicators — suppress hard brand filter for comparative queries
COMPARISON_INDICATORS: Set[str] = {
    "like", "style", "similar to", "inspired by", "alternatives to",
    "alternative to", "compatible with",
}

# Soft preference terms that stay in semantic query for ranking signals
SOFT_PREFERENCE_TERMS: Set[str] = {
    "budget", "affordable", "cheap", "premium", "luxury", "high-end", "best",
    "top-rated", "comfortable", "powerful", "lightweight", "portable", "compact",
    "durable", "wireless", "quiet", "ergonomic", "heavy-duty", "fast",
    "reliable", "cushion", "cushioning", "waterproof",
}

# Price-related operator words that should be stripped from semantic query
# after price extraction (in case regex captured the number but not the operator word)
PRICE_OPERATOR_WORDS: Set[str] = {
    "under", "below", "above", "over", "between", "less", "more", "than",
    "max", "maximum", "min", "minimum", "up", "least", "most",
}

6566                                                                               

# ============================================================================
# §2.1 — ParsedQuery Data Contract (Pydantic)
# ============================================================================

class PriceConstraint(BaseModel):
    """Extracted price constraint with character offsets for semantic cleanup."""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    raw_span: Tuple[int, int] = (0, 0)  # char offsets removed from original query


class TokenCorrection(BaseModel):
    """Per-token correction record with confidence and source field."""
    original: str
    corrected: str
    confidence: float = 0.0     # 0.0–1.0
    source_field: str = "uncorrected"  # brand | category | product_name | tag | description | uncorrected

    model_config = {"frozen": False}


class ParsedQuery(BaseModel):
    """Structured representation of a parsed search query.

    All downstream stages (fuzzy, semantic, exact, partial, ranking)
    consume this object — no stage re-parses the raw string.
    """
    raw_query: str = ""
    normalized_query: str = ""
    price: Optional[PriceConstraint] = None
    tokens: List[TokenCorrection] = Field(default_factory=list)
    semantic_query: str = ""
    is_explicit_product_query: bool = False
    detected_category_anchor: Optional[str] = None
    detected_brand_anchor: Optional[str] = None
    # Additional fields for backward compatibility and ranking
    detected_brands: List[str] = Field(default_factory=list)
    detected_categories: List[str] = Field(default_factory=list)
    normalized_query_variants: List[str] = Field(default_factory=list)
    soft_preferences: List[str] = Field(default_factory=list)
    is_brand_hard_filter: bool = False
    is_category_hard_filter: bool = False

    # Convenience properties for backward compatibility
    @property
    def original_query(self) -> str:
        return self.raw_query

    @property
    def min_price(self) -> Optional[float]:
        return self.price.min_price if self.price else None

    @property
    def max_price(self) -> Optional[float]:
        return self.price.max_price if self.price else None

    @property
    def normalized_semantic_query(self) -> str:
        return self.normalized_query_variants[0] if self.normalized_query_variants else self.semantic_query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "semantic_query": self.semantic_query,
            "price": self.price.model_dump() if self.price else None,
            "tokens": [t.model_dump() for t in self.tokens],
            "is_explicit_product_query": self.is_explicit_product_query,
            "detected_category_anchor": self.detected_category_anchor,
            "detected_brand_anchor": self.detected_brand_anchor,
            "detected_brands": self.detected_brands,
            "detected_categories": self.detected_categories,
            "normalized_query_variants": self.normalized_query_variants,
            "soft_preferences": self.soft_preferences,
            "is_brand_hard_filter": self.is_brand_hard_filter,
            "is_category_hard_filter": self.is_category_hard_filter,
            "min_price": self.min_price,
            "max_price": self.max_price,
        }


# ============================================================================
# Phonetic Helper
# ============================================================================

def _soundex(word: str) -> str:
    """Standard Soundex phonetic algorithm for spelling error recovery."""
    w = word.upper()
    if not w:
        return ""
    code_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    first_letter = w[0]
    codes = []
    prev_code = code_map.get(first_letter, '')
    for char in w[1:]:
        c = code_map.get(char, '')
        if c and c != prev_code:
            codes.append(c)
        prev_code = c
    return (first_letter + "".join(codes) + "0000")[:4]


# ============================================================================
# §3.1 — Catalog Vocabulary with Field-Separated Indexes
# ============================================================================

class _FieldVocabEntry:
    """Entry in a field-specific vocabulary index."""
    __slots__ = ('token', 'canonical', 'field', 'count', 'soundex')

    def __init__(self, token: str, canonical: str, field: str, count: int):
        self.token = token
        self.canonical = canonical
        self.field = field
        self.count = count
        self.soundex = _soundex(token)


class CatalogVocabulary:
    """
    In-memory singleton holding catalog vocabulary discovered dynamically from PostgreSQL.
    Zero hardcoded brand/category names.

    Key difference from v1: vocabulary is split into SEPARATE per-field indexes
    (brand_vocab, category_vocab, product_name_vocab, tag_vocab) so that brand
    corrections compete fairly against product-name corrections instead of one
    large pool drowning the other.
    """
    _instance: Optional["CatalogVocabulary"] = None

    def __init__(self):
        self.brands: List[str] = []
        self.categories: List[str] = []
        self.brand_lower_map: Dict[str, str] = {}
        self.category_lower_map: Dict[str, str] = {}
        # Field-separated vocab indexes (§3.1)
        self.brand_vocab: List[_FieldVocabEntry] = []
        self.category_vocab: List[_FieldVocabEntry] = []
        self.product_name_vocab: List[_FieldVocabEntry] = []
        self.tag_vocab: List[_FieldVocabEntry] = []
        # Global frequency for normalization
        self.max_frequency: int = 1
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "CatalogVocabulary":
        if cls._instance is None:
            cls._instance = CatalogVocabulary()
        return cls._instance

    def load(self, db: Session, force_reload: bool = False) -> None:
        if self._is_loaded and not force_reload:
            return

        # Fetch unique non-empty brands
        raw_brands = [
            b for b in db.scalars(select(distinct(Product.brand))).all() if b and b.strip()
        ]
        raw_categories = [
            c for c in db.scalars(select(distinct(Product.category))).all() if c and c.strip()
        ]

        self.brands = sorted(raw_brands)
        self.categories = sorted(raw_categories)
        self.brand_lower_map = {b.lower(): b for b in self.brands}
        self.category_lower_map = {c.lower(): c for c in self.categories}

        # Build field-separated vocabulary indexes
        brand_freq: Counter = Counter()
        category_freq: Counter = Counter()
        product_name_freq: Counter = Counter()
        tag_freq: Counter = Counter()

        products_data = db.execute(
            select(Product.product_name, Product.brand, Product.category, Product.tags, Product.description)
        ).fetchall()

        for row in products_data:
            pname, brand, category, tags, description = row

            if brand:
                for token in self._tokenize_field(brand):
                    brand_freq[token] += 1
            if category:
                for token in self._tokenize_field(category):
                    category_freq[token] += 1
            if pname:
                for token in self._tokenize_field(pname):
                    product_name_freq[token] += 1
            if tags:
                for token in self._tokenize_field(str(tags)):
                    tag_freq[token] += 1

        # Build vocab entries per field
        self.brand_vocab = self._build_vocab_entries(brand_freq, self.brands, "brand")
        self.category_vocab = self._build_vocab_entries(category_freq, self.categories, "category")
        self.product_name_vocab = self._build_pname_vocab(product_name_freq)
        self.tag_vocab = self._build_tag_vocab(tag_freq)

        # Global max frequency for normalization
        all_freqs = list(brand_freq.values()) + list(category_freq.values()) + \
                    list(product_name_freq.values()) + list(tag_freq.values())
        self.max_frequency = max(all_freqs) if all_freqs else 1

        self._is_loaded = True

    @staticmethod
    def _tokenize_field(text: str) -> List[str]:
        """Extract word tokens from a field value."""
        return [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', text) if len(w) >= 2]

    def _build_vocab_entries(
        self, freq: Counter, canonical_values: List[str], field: str
    ) -> List[_FieldVocabEntry]:
        """Build vocab entries for brands/categories.

        Each canonical value (e.g. 'Nike') contributes both itself as a whole
        and its individual tokens.
        """
        entries = []
        seen_tokens: Set[str] = set()

        for canonical in canonical_values:
            lower_canonical = canonical.lower()

            # Add whole-value entry
            if lower_canonical not in seen_tokens and len(lower_canonical) >= 2:
                seen_tokens.add(lower_canonical)
                count = sum(freq.get(t, 0) for t in self._tokenize_field(canonical))
                entries.append(_FieldVocabEntry(lower_canonical, canonical, field, max(count, 1)))

            # Add individual tokens
            for token in self._tokenize_field(canonical):
                if token not in seen_tokens and token not in STOP_WORDS:
                    seen_tokens.add(token)
                    entries.append(_FieldVocabEntry(token, canonical, field, freq.get(token, 1)))

        return entries

    def _build_pname_vocab(self, freq: Counter) -> List[_FieldVocabEntry]:
        """Build product-name token vocabulary."""
        entries = []
        for token, count in freq.items():
            if len(token) >= 3 and token not in STOP_WORDS:
                entries.append(_FieldVocabEntry(token, token, "product_name", count))
        return entries

    def _build_tag_vocab(self, freq: Counter) -> List[_FieldVocabEntry]:
        """Build tag token vocabulary."""
        entries = []
        for token, count in freq.items():
            if len(token) >= 3 and token not in STOP_WORDS:
                entries.append(_FieldVocabEntry(token, token, "tag", count))
        return entries

    # ==================================================================
    # §3.2 + §3.3 — Token Correction with Field Priority
    # ==================================================================

    def correct_token(self, token: str) -> List[TokenCorrection]:
        """Correct a single token against field-separated vocabularies.

        Returns a list of correction candidates sorted by confidence.
        Uses fuzz.ratio (full-string Levenshtein-based similarity) for
        single-token correction — NOT token_set_ratio.

        Phonetic matching is an INDEPENDENT candidate path, not gated
        behind Levenshtein distance.
        """
        lower_tok = token.lower()
        tok_len = len(lower_tok)

        # ≤2 chars: no correction unless exact catalog match
        if tok_len <= 2:
            exact = self._check_exact_match(lower_tok)
            if exact:
                return [TokenCorrection(
                    original=token, corrected=exact[0],
                    confidence=1.0, source_field=exact[1],
                )]
            return [TokenCorrection(original=token, corrected=token, confidence=0.0, source_field="uncorrected")]

        # Skip stop words and digits
        if lower_tok in STOP_WORDS or lower_tok.isdigit():
            return [TokenCorrection(original=token, corrected=token, confidence=0.0, source_field="uncorrected")]

        # Check exact match first (any field)
        exact = self._check_exact_match(lower_tok)
        if exact:
            return [TokenCorrection(
                original=token, corrected=exact[0],
                confidence=1.0, source_field=exact[1],
            )]

        # Collect candidates from all field vocabs
        tok_soundex = _soundex(lower_tok)
        candidates: List[TokenCorrection] = []

        # Search each field vocabulary separately
        for field_vocab, field_name in [
            (self.brand_vocab, "brand"),
            (self.category_vocab, "category"),
            (self.product_name_vocab, "product_name"),
            (self.tag_vocab, "tag"),
        ]:
            for entry in field_vocab:
                # Skip very short vocab entries when matching longer tokens
                if len(entry.token) < 3 and tok_len >= 3:
                    continue

                # Compute edit distance
                dist = distance.Levenshtein.distance(lower_tok, entry.token)

                # Determine max distance based on token length
                max_dist = MAX_EDIT_DISTANCE_SHORT if tok_len <= 4 else MAX_EDIT_DISTANCE_LONG

                # Phonetic match check (independent of Levenshtein)
                phonetic_match = (entry.soundex == tok_soundex) and len(lower_tok) >= 3

                # Candidate passes if within Levenshtein threshold OR has phonetic match
                if dist <= max_dist or (phonetic_match and dist <= MAX_EDIT_DISTANCE_PHONETIC):
                    # Use max(fuzz.ratio, JaroWinkler) for better length-mismatch handling (§3.2)
                    ratio_sim = fuzz.ratio(lower_tok, entry.token) / 100.0
                    jw_sim = distance.JaroWinkler.similarity(lower_tok, entry.token)
                    similarity = max(ratio_sim, jw_sim)

                    # Compute confidence (§3.3)
                    freq_score = math.log1p(entry.count) / math.log1p(self.max_frequency) if self.max_frequency > 0 else 0.0
                    field_bonus = FIELD_PRIORITY_BONUS.get(entry.field, 0.0)

                    confidence = (
                        W_SIM * similarity +
                        W_FREQ * freq_score +
                        W_FIELD_PRIORITY * field_bonus
                    )

                    # Phonetic match bonus
                    if phonetic_match:
                        confidence += 0.05

                    candidates.append(TokenCorrection(
                        original=token,
                        corrected=entry.canonical if entry.field in ("brand", "category") else entry.token,
                        confidence=min(confidence, 1.0),
                        source_field=entry.field,
                    ))

        if not candidates:
            return [TokenCorrection(original=token, corrected=token, confidence=0.0, source_field="uncorrected")]

        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        # Length-gated threshold check (§3.4)
        min_conf = MIN_CONFIDENCE_SHORT if tok_len <= 4 else MIN_CONFIDENCE_LONG
        if candidates[0].confidence < min_conf:
            return [TokenCorrection(original=token, corrected=token, confidence=0.0, source_field="uncorrected")]

        # Multi-candidate handling (§3.5): retain near-ties in different fields
        result = [candidates[0]]
        top_conf = candidates[0].confidence
        for c in candidates[1:]:
            if (top_conf - c.confidence) <= MULTI_CANDIDATE_MARGIN:
                # Only keep if it's from a different field (adds diversity)
                if c.source_field != result[0].source_field or c.corrected != result[0].corrected:
                    result.append(c)
                    if len(result) >= 3:
                        break
            else:
                break

        return result

    def _check_exact_match(self, lower_token: str) -> Optional[Tuple[str, str]]:
        """Check for exact match in brand or category maps."""
        if lower_token in self.brand_lower_map:
            return (self.brand_lower_map[lower_token], "brand")
        if lower_token in self.category_lower_map:
            return (self.category_lower_map[lower_token], "category")
        return None

    # ==================================================================
    # Brand / Category Detection
    # ==================================================================

    def find_matching_brand(self, token_or_phrase: str, is_single_token_query: bool = False) -> Optional[Tuple[str, float, bool]]:
        """Match a token or multi-word phrase against catalog brands.

        Returns (canonical_brand_name, confidence_score, is_exact_match) or None.
        Uses fuzz.ratio for single-token matching (§3.2).
        """
        if not self._is_loaded or not token_or_phrase:
            return None

        q = token_or_phrase.strip().lower()
        if not q or len(q) < 2:
            return None

        # 1. Exact match (case-insensitive)
        if q in self.brand_lower_map:
            return self.brand_lower_map[q], 100.0, True

        # Never fuzzy-match stop words, very short tokens, or exact category names/tokens against brands
        if q in STOP_WORDS or len(q) <= 2 or q in self.category_lower_map:
            return None

        # Check if q is a token in any category name (e.g. 'gaming' in 'Gaming', 'audio' in 'Audio')
        is_cat_token = any(
            q in [t.lower() for t in re.split(r'[\s&,/\-]+', cat)]
            for cat in self.category_lower_map
        )
        if is_cat_token:
            return None

        # 2. Fuzzy match using fuzz.ratio (§3.2 — single-token scorer)
        best_brand = None
        best_score = 0.0

        for lower_brand, canonical in self.brand_lower_map.items():
            if len(lower_brand) <= 2:
                continue

            dist = distance.Levenshtein.distance(q, lower_brand)
            ratio = float(fuzz.ratio(q, lower_brand))
            jw = float(distance.JaroWinkler.similarity(q, lower_brand)) * 100.0
            eff_similarity = max(ratio, jw)

            # Short brands (3-4 chars, e.g. "Nike", "Sony", "Puma", "Dell")
            if len(lower_brand) <= 4:
                if dist <= 1 or eff_similarity >= 75.0:
                    score = max(eff_similarity, 100.0 - (dist * 20.0))
                    if score > best_score:
                        best_score = score
                        best_brand = canonical
            else:
                # Longer brands (>= 5 chars, e.g. "Adidas", "Samsung", "Logitech")
                if (dist <= 1 and eff_similarity >= 75.0) or (dist <= 2 and eff_similarity >= 85.0):
                    if not is_single_token_query and len(q) <= 4 and dist > 0:
                        continue
                    if eff_similarity > best_score:
                        best_score = eff_similarity
                        best_brand = canonical

        if best_brand and best_score >= 75.0:
            return best_brand, best_score, False

        return None

    def find_matching_category(self, token_or_phrase: str) -> Optional[Tuple[str, float]]:
        """Match a token or phrase against catalog categories.

        Returns (canonical_category_name, confidence_score) or None.
        """
        if not self._is_loaded or not token_or_phrase:
            return None

        q = token_or_phrase.strip().lower()
        if not q or len(q) < 3 or q in STOP_WORDS:
            return None

        # 1. Exact match on full category name
        if q in self.category_lower_map:
            return self.category_lower_map[q], 100.0

        # 2. Token-level match against multi-word categories
        # e.g. "electronics" matches "Electronics", "audio" matches "Audio"
        for lower_cat, canonical in self.category_lower_map.items():
            cat_tokens = [t.lower() for t in re.split(r'[\s&,/\-]+', lower_cat) if len(t) >= 3]
            if q in cat_tokens:
                return canonical, 95.0

        # 3. High-confidence fuzzy match (e.g. "eletronics" -> "Electronics")
        best_cat = None
        best_score = 0.0
        for lower_cat, canonical in self.category_lower_map.items():
            ratio = float(fuzz.ratio(q, lower_cat))
            if ratio >= CATEGORY_MATCH_CONFIDENCE_MIN and ratio > best_score:
                best_score = ratio
                best_cat = canonical

        if best_cat:
            return best_cat, best_score

        return None


# ============================================================================
# §4 — Price Parser
# ============================================================================

def _normalize_amount(raw: str) -> float:
    """Normalize a price amount string to a float, handling k/K and 'thousand'."""
    raw = raw.strip().lower().replace(",", "")
    # Strip currency symbols and text prefixes
    raw = re.sub(r'^[₹$]|^rs\.?\s*|^inr\s*|^usd\s*', '', raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r'[₹$]|rs\.?|inr|usd', '', raw, flags=re.IGNORECASE).strip()
    if raw.endswith("k"):
        return float(raw[:-1]) * 1_000
    if "thousand" in raw:
        numeric_part = re.match(r'[\d.]+', raw)
        if numeric_part:
            return float(numeric_part.group()) * 1_000
    return float(raw)


# Currency prefix pattern for regex
_CURRENCY = r"(?:₹|rs\.?\s*|inr\s*|\$|usd\s*)?"
_CURRENCY_ALL = r"(?:₹|rs\.?\s*|inr\s*|\$|usd\s*)?"

# Number pattern that supports k/K suffix, commas, and 'thousand'
_NUMBER = r"[\d]+(?:[,.][\d]+)*\s*(?:k|K|thousand)?"

# Operators lists for fuzzy/typo matching
_UPPER_OPERATOR_WORDS = {"under", "unders", "undr", "below", "belw", "blo", "less", "les", "upto", "max", "maximum", "within", "sub", "cheaper"}
_LOWER_OPERATOR_WORDS = {"above", "abov", "abovee", "over", "ovr", "more", "min", "minimum", "higher", "starting", "from", "least"}
_RANGE_OPERATOR_WORDS = {"between", "btwn", "range", "from"}

# Price patterns — RANGE must be attempted BEFORE UNDER/ABOVE (§4.3)
_PRICE_PATTERNS = [
    # 1. Range: "between 300 and 700", "btwn 300 and 700", "₹300 to ₹700", "300-700", "300 to 700"
    (
        re.compile(
            rf"\b(?:between|btwn|from|range\s+of)?\s*{_CURRENCY}\s*({_NUMBER})\s*(?:and|to|\-)\s*{_CURRENCY}\s*({_NUMBER})\b",
            re.IGNORECASE,
        ),
        "range",
    ),
    # 2. Upper bound (prefix): "under 500", "unders 160", "undr 160", "below 2k", "less than 500", "up to 500", "upto 500", "max 500", "<= 500"
    (
        re.compile(
            rf"\b(?:under|unders|undr|below|belw|blo|less\s+than|les\s+than|lessthan|up\s+to|upto|max|maximum|at\s+most|atmost|within|sub|cheaper\s+than|budget\s+of|<=?)\s*{_CURRENCY}\s*({_NUMBER})\b",
            re.IGNORECASE,
        ),
        "max",
    ),
    # 3. Upper bound (suffix): "500 or less", "160 max", "160 budget", "500 and under"
    (
        re.compile(
            rf"\b{_CURRENCY}\s*({_NUMBER})\s*(?:or\s+less|or\s+below|max|maximum|budget|and\s+under)\b",
            re.IGNORECASE,
        ),
        "max",
    ),
    # 4. Lower bound (prefix): "above 500", "abov 500", "abovee 500", "over 10k", "ovr 10k", "more than 500", "min 500", ">= 500", "starting from 500"
    (
        re.compile(
            rf"\b(?:above|abov|abovee|over|ovr|more\s+than|morethan|min|minimum|at\s+least|atleast|higher\s+than|starting\s+from|from|>=?)\s*{_CURRENCY}\s*({_NUMBER})\b",
            re.IGNORECASE,
        ),
        "min",
    ),
    # 5. Lower bound (suffix): "500+", "500 and above", "500 or more", "500 min"
    (
        re.compile(
            rf"\b{_CURRENCY}\s*({_NUMBER})\s*(?:\+|and\s+above|and\s+over|or\s+more|min|minimum)\b",
            re.IGNORECASE,
        ),
        "min",
    ),
]


def _extract_price_constraint(text: str) -> Tuple[Optional[PriceConstraint], str]:
    """Extract price constraint from query text using regex patterns and dynamic fuzzy operator matching.

    Returns (PriceConstraint or None, remaining_text_with_price_span_removed).
    """
    # 1. First attempt structured regex matching
    for pattern, kind in _PRICE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                if kind == "range":
                    p1 = _normalize_amount(match.group(1))
                    p2 = _normalize_amount(match.group(2))
                    if p1 > p2 and PRICE_SWAP_ON_INVALID_RANGE:
                        p1, p2 = p2, p1
                    constraint = PriceConstraint(
                        min_price=p1, max_price=p2,
                        raw_span=(match.start(), match.end()),
                    )
                elif kind == "max":
                    val = _normalize_amount(match.group(1))
                    constraint = PriceConstraint(
                        max_price=val,
                        raw_span=(match.start(), match.end()),
                    )
                elif kind == "min":
                    val = _normalize_amount(match.group(1))
                    constraint = PriceConstraint(
                        min_price=val,
                        raw_span=(match.start(), match.end()),
                    )
                else:
                    continue

                # Remove the matched price span from the text
                remaining = text[:match.start()] + " " + text[match.end():]
                remaining = _strip_dangling_operators(remaining)
                remaining = re.sub(r"\s+", " ", remaining).strip()
                return constraint, remaining

            except (ValueError, IndexError):
                continue

    # 2. Dynamic Fuzzy Price Operator Scanner (catches typos like "unders 160", "undr 2k", "abov 500")
    words = text.split()
    for i, word in enumerate(words):
        # Check if word looks like a price number (e.g. "160", "2k", "$500", "₹1000", "1500")
        num_clean = re.sub(r'^[₹$]|^rs\.?\s*|^inr\s*|^usd\s*', '', word, flags=re.IGNORECASE)
        num_clean = re.sub(r'[₹$]|rs\.?|inr|usd', '', num_clean, flags=re.IGNORECASE)
        is_num = bool(re.match(r'^\d+(?:\.\d+)?(?:k|K|thousand)?$', num_clean))

        if is_num and i > 0:
            prev_word = words[i - 1].lower().strip(".,;:!?")
            # Check if prev_word fuzzy-matches an upper bound operator
            is_upper = any(
                distance.Levenshtein.distance(prev_word, op) <= 1 or fuzz.ratio(prev_word, op) >= 75
                for op in _UPPER_OPERATOR_WORDS
            )
            if is_upper:
                try:
                    val = _normalize_amount(num_clean)
                    # Find span in original text
                    pattern = re.compile(rf"\b{re.escape(words[i-1])}\s+{re.escape(word)}\b", re.IGNORECASE)
                    m = pattern.search(text)
                    span = (m.start(), m.end()) if m else (0, 0)
                    remaining = text[:span[0]] + " " + text[span[1]:] if m else " ".join(words[:i-1] + words[i+1:])
                    remaining = _strip_dangling_operators(remaining)
                    remaining = re.sub(r"\s+", " ", remaining).strip()
                    return PriceConstraint(max_price=val, raw_span=span), remaining
                except Exception:
                    pass

            # Check if prev_word fuzzy-matches a lower bound operator
            is_lower = any(
                distance.Levenshtein.distance(prev_word, op) <= 1 or fuzz.ratio(prev_word, op) >= 75
                for op in _LOWER_OPERATOR_WORDS
            )
            if is_lower:
                try:
                    val = _normalize_amount(num_clean)
                    pattern = re.compile(rf"\b{re.escape(words[i-1])}\s+{re.escape(word)}\b", re.IGNORECASE)
                    m = pattern.search(text)
                    span = (m.start(), m.end()) if m else (0, 0)
                    remaining = text[:span[0]] + " " + text[span[1]:] if m else " ".join(words[:i-1] + words[i+1:])
                    remaining = _strip_dangling_operators(remaining)
                    remaining = re.sub(r"\s+", " ", remaining).strip()
                    return PriceConstraint(min_price=val, raw_span=span), remaining
                except Exception:
                    pass

    return None, text


def _strip_dangling_operators(text: str) -> str:
    """Remove isolated price operator words left after price number extraction."""
    tokens = text.split()
    cleaned = []
    for tok in tokens:
        lower_tok = tok.lower().strip(".,;:!?")
        # Only remove if it's a pure operator word (not part of a brand like "Under Armour")
        if lower_tok in PRICE_OPERATOR_WORDS and len(tok) == len(lower_tok):
            continue
        cleaned.append(tok)
    return " ".join(cleaned)


# ============================================================================
# §7 — Semantic Query Construction
# ============================================================================

def _build_semantic_query(remaining_text: str) -> str:
    """Build clean semantic query from remaining text after price extraction.

    Keeps descriptive/preference adjectives. Strips bare comparison operators
    and dangling punctuation.
    """
    if not remaining_text or not remaining_text.strip():
        return ""

    # Remove bare comparison operators left after price extraction
    cleaned = remaining_text.strip()
    # Remove standalone < > <= >= symbols
    cleaned = re.sub(r'\s*[<>]=?\s*', ' ', cleaned)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ============================================================================
# Main Parse Function
# ============================================================================

def parse_query(db: Session, raw_query: str) -> ParsedQuery:
    """Parse a raw user search query into a structured ParsedQuery.

    This is the single entry point. All downstream stages consume
    the returned ParsedQuery — no stage re-parses the raw string.
    """
    cleaned_original = raw_query.strip()
    if not cleaned_original:
        return ParsedQuery(raw_query="", semantic_query="")

    vocab = CatalogVocabulary.get_instance()
    vocab.load(db)

    # ===================================================================
    # Step 1: Extract Price Constraints & Strip Price Phrase (§4)
    # ===================================================================
    price_constraint, remaining_text = _extract_price_constraint(cleaned_original)

    # Build the semantic query (price-stripped, operator-cleaned)
    semantic_query = _build_semantic_query(remaining_text)
    if not semantic_query:
        semantic_query = cleaned_original

    # ===================================================================
    # Step 2: Tokenize & Correct Tokens (§3)
    # ===================================================================
    raw_tokens = [t for t in re.split(r'[\s,\-/]+', semantic_query) if t and len(t) >= 1]
    corrected_tokens: List[TokenCorrection] = []
    normalized_parts: List[str] = []

    for tok in raw_tokens:
        corrections = vocab.correct_token(tok)
        corrected_tokens.extend(corrections)
        # Use the top correction for the normalized query
        if corrections and corrections[0].source_field != "uncorrected":
            normalized_parts.append(corrections[0].corrected)
        else:
            normalized_parts.append(tok)

    # Build normalized query variants from multi-candidate corrections
    primary_variant = " ".join(normalized_parts)
    variants = [primary_variant]

    # Generate alternative variants from OR'd candidates (§3.5)
    # Only generate a few to avoid combinatorial explosion
    alt_parts = list(normalized_parts)
    for i, tok in enumerate(raw_tokens):
        corrections = vocab.correct_token(tok)
        if len(corrections) > 1 and corrections[1].source_field != "uncorrected":
            alt_parts_copy = list(normalized_parts)
            alt_parts_copy[i] = corrections[1].corrected
            alt_variant = " ".join(alt_parts_copy)
            if alt_variant not in variants:
                variants.append(alt_variant)
                if len(variants) >= 4:
                    break

    # ===================================================================
    # Step 3: Detect Brands & Categories (§5)
    # ===================================================================
    detected_brands: List[str] = []
    detected_categories: List[str] = []
    detected_brand_anchor: Optional[str] = None
    detected_category_anchor: Optional[str] = None
    has_exact_brand_match = False

    # Check multi-word windows (1 to 3 words) for brand matching
    search_tokens = [t for t in re.split(r'[\s,\-/]+', semantic_query) if t]
    is_single_word = len(search_tokens) == 1
    matched_brand_names: Set[str] = set()

    for n in range(min(3, len(search_tokens)), 0, -1):
        for i in range(len(search_tokens) - n + 1):
            window = " ".join(search_tokens[i:i + n])
            match_res = vocab.find_matching_brand(window, is_single_token_query=is_single_word)
            if match_res:
                brand_name, confidence, is_exact = match_res
                if is_exact:
                    has_exact_brand_match = True
                if brand_name not in matched_brand_names:
                    matched_brand_names.add(brand_name)
                    detected_brands.append(brand_name)
                    if detected_brand_anchor is None:
                        detected_brand_anchor = brand_name

    # Also check corrected tokens for brand matches
    for tc in corrected_tokens:
        if tc.source_field == "brand" and tc.corrected not in matched_brand_names:
            matched_brand_names.add(tc.corrected)
            detected_brands.append(tc.corrected)
            if detected_brand_anchor is None:
                detected_brand_anchor = tc.corrected

    # Check tokens and full query for category matching
    matched_cat_names: Set[str] = set()
    for n in range(min(3, len(search_tokens)), 0, -1):
        for i in range(len(search_tokens) - n + 1):
            window = " ".join(search_tokens[i:i + n])
            cat_res = vocab.find_matching_category(window)
            if cat_res:
                cat_name, confidence = cat_res
                if cat_name not in matched_cat_names:
                    matched_cat_names.add(cat_name)
                    detected_categories.append(cat_name)
                    if detected_category_anchor is None and confidence >= CATEGORY_MATCH_CONFIDENCE_MIN:
                        detected_category_anchor = cat_name

    # Also check corrected tokens for category matches
    for tc in corrected_tokens:
        if tc.source_field == "category" and tc.corrected not in matched_cat_names:
            matched_cat_names.add(tc.corrected)
            detected_categories.append(tc.corrected)
            if detected_category_anchor is None:
                detected_category_anchor = tc.corrected

    # ===================================================================
    # Step 4: Comparison Indicators (§5) & Hard-Filter Safety
    # ===================================================================
    lower_query = cleaned_original.lower()
    is_comparative = any(comp in lower_query for comp in COMPARISON_INDICATORS)

    # Brand hard-filter applies when:
    # 1. Non-comparative query AND
    # 2. Either has an exact brand match (e.g. "Nike shoes") OR is a single-token brand search (e.g. "nykee")
    is_brand_hard_filter = bool(
        detected_brands and not is_comparative and (has_exact_brand_match or is_single_word)
    )
    is_category_hard_filter = bool(detected_category_anchor and not is_comparative and is_single_word)
    is_explicit_product_query = bool(detected_category_anchor)

    # ===================================================================
    # Step 5: Detect Soft Preferences
    # ===================================================================
    detected_soft_preferences = [
        pref for pref in SOFT_PREFERENCE_TERMS if pref in lower_query
    ]

    return ParsedQuery(
        raw_query=cleaned_original,
        normalized_query=primary_variant,
        price=price_constraint,
        tokens=corrected_tokens,
        semantic_query=semantic_query,
        is_explicit_product_query=is_explicit_product_query,
        detected_category_anchor=detected_category_anchor,
        detected_brand_anchor=detected_brand_anchor,
        detected_brands=detected_brands,
        detected_categories=detected_categories,
        normalized_query_variants=variants,
        soft_preferences=detected_soft_preferences,
        is_brand_hard_filter=is_brand_hard_filter,
        is_category_hard_filter=is_category_hard_filter,
    )
