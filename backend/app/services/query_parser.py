"""
Dynamic Query Understanding & Parsing Module — Hardened v2.1

Architecture & Design Principles:
---------------------------------
1. Generic NLP Parsing (Zero hardcoded brands, categories, or typo dictionaries).
2. Dynamic Vocabulary: Discovered at runtime from PostgreSQL, with SEPARATE
   per-field indexes (brand, category, product_name, tag) and dynamic product-type
   to category mappings.
3. Clean Separation of Stages:
   Raw Query -> Price Extraction -> Tokenization -> Token Correction ->
   Normalized Query / "Did you mean" -> Entity Detection on Normalized Tokens ->
   Intent & Preference Detection -> ParsedQuery.
4. Glued Stopword / Preposition Handling (e.g., 'oflogitch' -> 'Logitech', 'thephone' -> 'phone').
5. High-Confidence Entity Gating: Never outputs random or low-confidence brands/categories.
6. Full pipeline returns ParsedQuery Pydantic model consumed by all downstream search stages.
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

# Glued prefixes that may accidentally be joined to words in user typos
GLUED_PREFIXES: Tuple[str, ...] = ("of", "the", "for", "in", "to", "by", "a", "an", "on", "at", "with", "from")

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


# ============================================================================
# §2.1 — ParsedQuery Data Contract (Pydantic)
# ============================================================================

class PriceConstraint(BaseModel):
    """Extracted price constraint with character offsets for semantic cleanup."""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    raw_span: Tuple[int, int] = (0, 0)  # char offsets removed from original query


class TokenCorrection(BaseModel):
    """Per-token correction record with confidence, similarity, and source field."""
    original: str
    corrected: str
    confidence: float = 0.0     # 0.0–1.0
    source_field: str = "uncorrected"  # brand | category | product_name | tag | description | uncorrected
    similarity: float = 0.0
    phonetic_match: bool = False

    model_config = {"frozen": False}


class ParsedQuery(BaseModel):
    """Structured representation of a parsed search query.

    All downstream stages (fuzzy, semantic, exact, partial, ranking)
    consume this object — no stage re-parses the raw string.
    """
    raw_query: str = ""
    normalized_query: str = ""
    did_you_mean: Optional[str] = None
    price: Optional[PriceConstraint] = None
    tokens: List[TokenCorrection] = Field(default_factory=list)
    semantic_query: str = ""
    is_explicit_product_query: bool = False
    detected_category_anchor: Optional[str] = None
    detected_brand_anchor: Optional[str] = None
    # Additional fields for ranking and interpretation
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
            "did_you_mean": self.did_you_mean,
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
    """
    _instance: Optional["CatalogVocabulary"] = None

    def __init__(self):
        self.brands: List[str] = []
        self.categories: List[str] = []
        self.brand_lower_map: Dict[str, str] = {}
        self.category_lower_map: Dict[str, str] = {}
        # Dynamic mapping from product-type tokens to their dominant category
        self.product_type_to_category: Dict[str, str] = {}
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
        product_token_cats = defaultdict(Counter)

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
                tokens = self._tokenize_field(pname)
                for token in tokens:
                    product_name_freq[token] += 1
                    if category and len(token) >= 3 and token not in STOP_WORDS:
                        product_token_cats[token][category] += 1
            if tags:
                tokens = self._tokenize_field(str(tags))
                for token in tokens:
                    tag_freq[token] += 1
                    if category and len(token) >= 3 and token not in STOP_WORDS:
                        product_token_cats[token][category] += 1

        # Dynamic product-type to category mapping (e.g. 'mouse' -> 'Electronics')
        self.product_type_to_category = {}
        for token, cat_counter in product_token_cats.items():
            top_cat, top_count = cat_counter.most_common(1)[0]
            total_count = sum(cat_counter.values())
            if (top_count / float(total_count) >= 0.50) and top_count >= 2:
                self.product_type_to_category[token] = top_cat

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

    def is_valid_word(self, token: str) -> bool:
        """Check if a word/token is a recognized valid catalog entity or stopword."""
        lower = token.lower().strip()
        if not lower:
            return False
        if lower in STOP_WORDS or lower in self.brand_lower_map or lower in self.category_lower_map:
            return True
        for entry in self.brand_vocab:
            if entry.token == lower:
                return True
        for entry in self.category_vocab:
            if entry.token == lower:
                return True
        for entry in self.product_name_vocab:
            if entry.token == lower:
                return True
        for entry in self.tag_vocab:
            if entry.token == lower:
                return True
        return False

    @staticmethod
    def _tokenize_field(text: str) -> List[str]:
        """Extract word tokens from a field value."""
        return [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', text) if len(w) >= 2]

    def _build_vocab_entries(
        self, freq: Counter, canonical_values: List[str], field: str
    ) -> List[_FieldVocabEntry]:
        """Build vocab entries for brands/categories."""
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
                    token_canon = canonical if token == lower_canonical else token
                    entries.append(_FieldVocabEntry(token, token_canon, field, freq.get(token, 1)))

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
    # §3.2 + §3.3 — Token Correction with Strict Confidence & Preposition Handling
    # ==================================================================

    def correct_token(self, token: str) -> List[TokenCorrection]:
        """Correct a single token against field-separated vocabularies.

        Handles:
        1. Exact matches across brands/categories/products.
        2. Glued prepositions (e.g. 'oflogitch' -> 'Logitech').
        3. Damerau-Levenshtein, RapidFuzz ratio, Jaro-Winkler, and Soundex.
        4. Strict length-gated thresholds so weak matches remain uncorrected.
        """
        lower_tok = token.lower().strip()
        tok_len = len(lower_tok)

        # ≤2 chars: no correction unless exact catalog match
        if tok_len <= 2:
            exact = self._check_exact_match(lower_tok)
            if exact:
                return [TokenCorrection(
                    original=token, corrected=exact[0],
                    confidence=1.0, source_field=exact[1], similarity=1.0,
                )]
            return [TokenCorrection(original=token, corrected=token, confidence=0.0, source_field="uncorrected", similarity=0.0)]

        # Skip stop words and digits
        if lower_tok in STOP_WORDS or lower_tok.isdigit():
            return [TokenCorrection(original=token, corrected=token, confidence=0.0, source_field="uncorrected", similarity=0.0)]

        # Check exact match first (any field)
        exact = self._check_exact_match(lower_tok)
        if exact:
            return [TokenCorrection(
                original=token, corrected=exact[0],
                confidence=1.0, source_field=exact[1], similarity=1.0,
            )]

        # Check exact match in product name / tag vocabs
        for entry in self.product_name_vocab:
            if entry.token == lower_tok:
                return [TokenCorrection(
                    original=token, corrected=token,
                    confidence=1.0, source_field="product_name", similarity=1.0,
                )]
        for entry in self.tag_vocab:
            if entry.token == lower_tok:
                return [TokenCorrection(
                    original=token, corrected=token,
                    confidence=1.0, source_field="tag", similarity=1.0,
                )]

        # Check glued preposition/stopword prefix (e.g. 'oflogitch' -> 'logitch')
        stripped_tok = None
        for pfx in GLUED_PREFIXES:
            if lower_tok.startswith(pfx) and len(lower_tok) - len(pfx) >= 3:
                candidate_sub = lower_tok[len(pfx):]
                # If stripped sub-token is exact match
                sub_exact = self._check_exact_match(candidate_sub)
                if sub_exact:
                    return [TokenCorrection(
                        original=token, corrected=sub_exact[0],
                        confidence=0.95, source_field=sub_exact[1], similarity=0.95,
                    )]
                stripped_tok = candidate_sub
                break

        # Collect candidates from all field vocabs
        tok_soundex = _soundex(lower_tok)
        col_tok = re.sub(r'(.)\1+', r'\1', lower_tok)
        col_soundex = _soundex(col_tok)
        stripped_soundex = _soundex(stripped_tok) if stripped_tok else None

        candidates: List[TokenCorrection] = []

        # Search each field vocabulary separately
        for field_vocab, field_name in [
            (self.brand_vocab, "brand"),
            (self.category_vocab, "category"),
            (self.product_name_vocab, "product_name"),
            (self.tag_vocab, "tag"),
        ]:
            for entry in field_vocab:
                if len(entry.token) < 3 and tok_len >= 3:
                    continue

                # Compute edit distance on raw, repeat-collapsed, and stripped tokens
                dist_raw = distance.DamerauLevenshtein.distance(lower_tok, entry.token)
                dist_col = distance.DamerauLevenshtein.distance(col_tok, entry.token)
                dist_strip = distance.DamerauLevenshtein.distance(stripped_tok, entry.token) if stripped_tok else 999
                dist = min(dist_raw, dist_col, dist_strip)

                # Determine max distance based on token length
                max_dist = MAX_EDIT_DISTANCE_SHORT if tok_len <= 4 else MAX_EDIT_DISTANCE_LONG
                # For longer tokens (>= 7 chars), allow distance 3 if stripped prefix or partial match is strong
                if tok_len >= 7 and (dist_strip <= 2 or fuzz.partial_ratio(lower_tok, entry.token) >= 85):
                    max_dist = 3

                # Phonetic match check
                phonetic_match = (
                    entry.soundex == tok_soundex or
                    entry.soundex == col_soundex or
                    (stripped_soundex is not None and entry.soundex == stripped_soundex)
                ) and len(lower_tok) >= 3

                if dist <= max_dist or (phonetic_match and dist <= MAX_EDIT_DISTANCE_PHONETIC):
                    ratio_raw = fuzz.ratio(lower_tok, entry.token) / 100.0
                    ratio_col = fuzz.ratio(col_tok, entry.token) / 100.0
                    ratio_strip = (fuzz.ratio(stripped_tok, entry.token) / 100.0) if stripped_tok else 0.0
                    ratio_sim = max(ratio_raw, ratio_col, ratio_strip)

                    jw_raw = distance.JaroWinkler.similarity(lower_tok, entry.token)
                    jw_col = distance.JaroWinkler.similarity(col_tok, entry.token)
                    jw_strip = distance.JaroWinkler.similarity(stripped_tok, entry.token) if stripped_tok else 0.0
                    jw_sim = max(jw_raw, jw_col, jw_strip)

                    eff_len = len(stripped_tok) if (stripped_tok and dist == dist_strip) else tok_len
                    lev_sim = max(0.0, 1.0 - (float(dist) / max(eff_len, len(entry.token))))
                    similarity = (0.40 * lev_sim) + (0.30 * ratio_sim) + (0.30 * jw_sim)

                    # Compute confidence with balanced weights
                    freq_score = math.log1p(entry.count) / math.log1p(self.max_frequency) if self.max_frequency > 0 else 0.0
                    field_bonus = FIELD_PRIORITY_BONUS.get(entry.field, 0.50)

                    confidence = (
                        W_SIM * similarity +
                        W_FREQ * freq_score +
                        W_FIELD_PRIORITY * field_bonus
                    )

                    if phonetic_match:
                        confidence += 0.05

                    candidates.append(TokenCorrection(
                        original=token,
                        corrected=entry.canonical if entry.field in ("brand", "category") else entry.token,
                        confidence=min(confidence, 1.0),
                        source_field=entry.field,
                        similarity=round(similarity, 4),
                        phonetic_match=phonetic_match,
                    ))

        if not candidates:
            return [TokenCorrection(original=token, corrected=token, confidence=0.0, source_field="uncorrected", similarity=0.0)]

        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        # Length-gated threshold check (§3.4)
        min_conf = MIN_CONFIDENCE_SHORT if tok_len <= 4 else MIN_CONFIDENCE_LONG
        if candidates[0].confidence < min_conf:
            return [TokenCorrection(original=token, corrected=token, confidence=0.0, source_field="uncorrected", similarity=0.0)]

        # Retain near-tie candidates for variants
        result = [candidates[0]]
        top_conf = candidates[0].confidence
        for c in candidates[1:]:
            if (top_conf - c.confidence) <= MULTI_CANDIDATE_MARGIN:
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
    # Brand / Category Detection (Strict Confidence Gated)
    # ==================================================================

    def find_matching_brand(self, token_or_phrase: str, min_confidence: float = 80.0) -> Optional[Tuple[str, float, bool]]:
        """Match a token or multi-word phrase against catalog brands.

        Returns (canonical_brand_name, confidence_score, is_exact_match) or None.
        """
        if not self._is_loaded or not token_or_phrase:
            return None

        q = token_or_phrase.strip().lower()
        if not q or len(q) < 2 or q in STOP_WORDS or q in self.category_lower_map:
            return None

        # 1. Exact match (case-insensitive)
        if q in self.brand_lower_map:
            return self.brand_lower_map[q], 100.0, True

        # Check if q is a token in any category name
        is_cat_token = any(
            q in [t.lower() for t in re.split(r'[\s&,/\-]+', cat)]
            for cat in self.category_lower_map
        )
        if is_cat_token:
            return None

        # 2. Fuzzy match against catalog brands
        best_brand = None
        best_score = 0.0

        for lower_brand, canonical in self.brand_lower_map.items():
            if len(lower_brand) <= 2:
                continue

            dist = distance.Levenshtein.distance(q, lower_brand)
            ratio = float(fuzz.ratio(q, lower_brand))
            jw = float(distance.JaroWinkler.similarity(q, lower_brand)) * 100.0
            eff_similarity = max(ratio, jw)

            # Short brands (3-4 chars, e.g. "Nike", "Sony", "Dell", "Puma")
            if len(lower_brand) <= 4:
                if dist <= 1 and eff_similarity >= 80.0:
                    score = max(eff_similarity, 100.0 - (dist * 15.0))
                    if score > best_score:
                        best_score = score
                        best_brand = canonical
            else:
                # Longer brands (>= 5 chars, e.g. "Logitech", "Samsung", "Adidas")
                if (dist <= 1 and eff_similarity >= 80.0) or (dist <= 2 and ratio >= 80.0 and eff_similarity >= 85.0):
                    if eff_similarity > best_score:
                        best_score = eff_similarity
                        best_brand = canonical

        if best_brand and best_score >= min_confidence:
            return best_brand, best_score, False

        return None

    def find_matching_category(self, token_or_phrase: str, min_confidence: float = 80.0) -> Optional[Tuple[str, float]]:
        """Match a token or phrase against catalog categories and dynamic product-type mappings.

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
        for lower_cat, canonical in self.category_lower_map.items():
            cat_tokens = [t.lower() for t in re.split(r'[\s&,/\-]+', lower_cat) if len(t) >= 3]
            if q in cat_tokens:
                return canonical, 95.0

        # 3. Dynamic Product-Type to Category Mapping (e.g. 'mouse' -> 'Electronics')
        if q in self.product_type_to_category:
            return self.product_type_to_category[q], 90.0

        # 4. High-confidence fuzzy match (e.g. "eletronics" -> "Electronics")
        best_cat = None
        best_score = 0.0
        for lower_cat, canonical in self.category_lower_map.items():
            ratio = float(fuzz.ratio(q, lower_cat))
            if ratio >= CATEGORY_MATCH_CONFIDENCE_MIN and ratio > best_score:
                best_score = ratio
                best_cat = canonical

        if best_cat and best_score >= min_confidence:
            return best_cat, best_score

        return None


# ============================================================================
# §4 — Price Parser
# ============================================================================

def _normalize_amount(raw: str) -> float:
    """Normalize a price amount string to a float, handling k/K and 'thousand'."""
    raw = raw.strip().lower().replace(",", "")
    raw = re.sub(r'^[₹$]|^rs\.?\s*|^inr\s*|^usd\s*', '', raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r'[₹$]|rs\.?|inr|usd', '', raw, flags=re.IGNORECASE).strip()
    if raw.endswith("k"):
        return float(raw[:-1]) * 1_000
    if "thousand" in raw:
        numeric_part = re.match(r'[\d.]+', raw)
        if numeric_part:
            return float(numeric_part.group()) * 1_000
    return float(raw)


_CURRENCY = r"(?:₹|rs\.?\s*|inr\s*|\$|usd\s*)?"
_NUMBER = r"[\d]+(?:[,.][\d]+)*\s*(?:k|K|thousand)?"

_UPPER_OPERATOR_WORDS = {"under", "unders", "undr", "below", "belw", "blo", "less", "les", "upto", "max", "maximum", "within", "sub", "cheaper"}
_LOWER_OPERATOR_WORDS = {"above", "abov", "abovee", "over", "ovr", "more", "min", "minimum", "higher", "starting", "from", "least"}

_PRICE_PATTERNS = [
    # 1. Range
    (
        re.compile(
            rf"\b(?:between|btwn|from|range\s+of)?\s*{_CURRENCY}\s*({_NUMBER})\s*(?:and|to|\-)\s*{_CURRENCY}\s*({_NUMBER})\b",
            re.IGNORECASE,
        ),
        "range",
    ),
    # 2. Upper bound (prefix)
    (
        re.compile(
            rf"\b(?:under|unders|undr|below|belw|blo|less\s+than|les\s+than|lessthan|up\s+to|upto|max|maximum|at\s+most|atmost|within|sub|cheaper\s+than|budget\s+of|<=?)\s*{_CURRENCY}\s*({_NUMBER})\b",
            re.IGNORECASE,
        ),
        "max",
    ),
    # 3. Upper bound (suffix)
    (
        re.compile(
            rf"\b{_CURRENCY}\s*({_NUMBER})\s*(?:or\s+less|or\s+below|max|maximum|budget|and\s+under)\b",
            re.IGNORECASE,
        ),
        "max",
    ),
    # 4. Lower bound (prefix)
    (
        re.compile(
            rf"\b(?:above|abov|abovee|over|ovr|more\s+than|morethan|min|minimum|at\s+least|atleast|higher\s+than|starting\s+from|from|>=?)\s*{_CURRENCY}\s*({_NUMBER})\b",
            re.IGNORECASE,
        ),
        "min",
    ),
    # 5. Lower bound (suffix)
    (
        re.compile(
            rf"\b{_CURRENCY}\s*({_NUMBER})\s*(?:\+|and\s+above|and\s+over|or\s+more|min|minimum)\b",
            re.IGNORECASE,
        ),
        "min",
    ),
]


def _extract_price_constraint(text: str) -> Tuple[Optional[PriceConstraint], str]:
    """Extract price constraint from query text using regex patterns and fuzzy operator matching."""
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

                remaining = text[:match.start()] + " " + text[match.end():]
                remaining = _strip_dangling_operators(remaining)
                remaining = re.sub(r"\s+", " ", remaining).strip()
                return constraint, remaining

            except (ValueError, IndexError):
                continue

    # Dynamic Fuzzy Price Operator Scanner
    words = text.split()
    for i, word in enumerate(words):
        num_clean = re.sub(r'^[₹$]|^rs\.?\s*|^inr\s*|^usd\s*', '', word, flags=re.IGNORECASE)
        num_clean = re.sub(r'[₹$]|rs\.?|inr|usd', '', num_clean, flags=re.IGNORECASE)
        is_num = bool(re.match(r'^\d+(?:\.\d+)?(?:k|K|thousand)?$', num_clean))

        if is_num and i > 0:
            prev_word = words[i - 1].lower().strip(".,;:!?")
            is_upper = any(
                distance.Levenshtein.distance(prev_word, op) <= 1 or fuzz.ratio(prev_word, op) >= 75
                for op in _UPPER_OPERATOR_WORDS
            )
            if is_upper:
                try:
                    val = _normalize_amount(num_clean)
                    pattern = re.compile(rf"\b{re.escape(words[i-1])}\s+{re.escape(word)}\b", re.IGNORECASE)
                    m = pattern.search(text)
                    span = (m.start(), m.end()) if m else (0, 0)
                    remaining = text[:span[0]] + " " + text[span[1]:] if m else " ".join(words[:i-1] + words[i+1:])
                    remaining = _strip_dangling_operators(remaining)
                    remaining = re.sub(r"\s+", " ", remaining).strip()
                    return PriceConstraint(max_price=val, raw_span=span), remaining
                except Exception:
                    pass

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
        if lower_tok in PRICE_OPERATOR_WORDS and len(tok) == len(lower_tok):
            continue
        cleaned.append(tok)
    return " ".join(cleaned)


def _build_semantic_query(remaining_text: str) -> str:
    """Build clean semantic query from remaining text after price extraction."""
    if not remaining_text or not remaining_text.strip():
        return ""
    cleaned = remaining_text.strip()
    cleaned = re.sub(r'\s*[<>]=?\s*', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ============================================================================
# Main Parse Function (Clean Separation of Correction & Entity Detection)
# ============================================================================

def parse_query(db: Session, raw_query: str) -> ParsedQuery:
    """Parse a raw user search query into a structured ParsedQuery.

    Pipeline:
    1. Extract Price -> Produces semantic_query without price span.
    2. Tokenize & Typo Correct -> Produces high-confidence normalized query.
    3. Detect Entities (Brand, Category, Preference) strictly on normalized tokens.
    4. Guard with confidence gating.
    """
    cleaned_original = raw_query.strip()
    if not cleaned_original:
        return ParsedQuery(raw_query="", semantic_query="")

    vocab = CatalogVocabulary.get_instance()
    vocab.load(db)

    # -------------------------------------------------------------------
    # Step 1: Price Constraints & Semantic Query
    # -------------------------------------------------------------------
    price_constraint, remaining_text = _extract_price_constraint(cleaned_original)
    base_semantic_text = _build_semantic_query(remaining_text)
    if not base_semantic_text:
        base_semantic_text = cleaned_original

    # -------------------------------------------------------------------
    # Step 2: Token Typo Correction
    # -------------------------------------------------------------------
    raw_tokens = [t for t in re.split(r'[\s,\-/]+', base_semantic_text) if t and len(t) >= 1]
    corrected_tokens: List[TokenCorrection] = []
    normalized_parts: List[str] = []
    has_real_correction = False

    for tok in raw_tokens:
        corrections = vocab.correct_token(tok)
        corrected_tokens.extend(corrections)
        top = corrections[0]
        if top.source_field != "uncorrected" and top.confidence >= 0.70:
            normalized_parts.append(top.corrected)
            if top.corrected.lower() != tok.lower() and top.confidence >= 0.75:
                has_real_correction = True
        else:
            normalized_parts.append(tok)

    normalized_query = " ".join(normalized_parts)

    # Set "Did you mean" only when meaningful high-confidence correction occurred
    did_you_mean = normalized_query if (has_real_correction and normalized_query.lower() != base_semantic_text.lower()) else None

    # Primary semantic search text uses normalized query if corrected, else base text
    effective_semantic_query = normalized_query if has_real_correction else base_semantic_text

    # Build variants for ChromaDB/fuzzy candidate retrieval
    variants = [effective_semantic_query]
    if base_semantic_text not in variants:
        variants.append(base_semantic_text)

    # -------------------------------------------------------------------
    # Step 3: Entity Detection on CORRECTED Tokens
    # -------------------------------------------------------------------
    detected_brands: List[str] = []
    detected_categories: List[str] = []
    detected_brand_anchor: Optional[str] = None
    detected_category_anchor: Optional[str] = None
    has_exact_brand_match = False

    # 3A. First inspect high-confidence token corrections
    for tc in corrected_tokens:
        if tc.source_field == "brand" and tc.confidence >= 0.75:
            if tc.corrected not in detected_brands:
                detected_brands.append(tc.corrected)
                if detected_brand_anchor is None:
                    detected_brand_anchor = tc.corrected
            if tc.confidence == 1.0 or tc.original.lower() == tc.corrected.lower():
                has_exact_brand_match = True

        if tc.source_field == "category" and tc.confidence >= 0.75:
            if tc.corrected not in detected_categories:
                detected_categories.append(tc.corrected)
                if detected_category_anchor is None:
                    detected_category_anchor = tc.corrected

    # 3B. Multi-word sliding window check over NORMALIZED tokens
    norm_tokens = [t for t in re.split(r'[\s,\-/]+', normalized_query) if t]
    for n in range(min(3, len(norm_tokens)), 0, -1):
        for i in range(len(norm_tokens) - n + 1):
            window = " ".join(norm_tokens[i:i + n])
            
            # Brand matching
            brand_res = vocab.find_matching_brand(window, min_confidence=80.0)
            if brand_res:
                bname, bconf, is_exact = brand_res
                if bname not in detected_brands:
                    detected_brands.append(bname)
                    if detected_brand_anchor is None:
                        detected_brand_anchor = bname
                if is_exact:
                    has_exact_brand_match = True

            # Category matching
            cat_res = vocab.find_matching_category(window, min_confidence=80.0)
            if cat_res:
                cname, cconf = cat_res
                if cname not in detected_categories:
                    detected_categories.append(cname)
                    if detected_category_anchor is None:
                        detected_category_anchor = cname

    # -------------------------------------------------------------------
    # Step 4: Comparison & Hard-Filter Safety
    # -------------------------------------------------------------------
    lower_query = cleaned_original.lower()
    is_comparative = any(comp in lower_query for comp in COMPARISON_INDICATORS)
    is_single_word = len(norm_tokens) == 1

    is_brand_hard_filter = bool(
        detected_brands and not is_comparative and (has_exact_brand_match or is_single_word)
    )
    is_category_hard_filter = bool(
        detected_category_anchor and not is_comparative and is_single_word
    )
    is_explicit_product_query = bool(detected_category_anchor)

    # -------------------------------------------------------------------
    # Step 5: Soft Preferences
    # -------------------------------------------------------------------
    detected_soft_preferences = [
        pref for pref in SOFT_PREFERENCE_TERMS
        if pref in normalized_query.lower() or pref in lower_query
    ]

    return ParsedQuery(
        raw_query=cleaned_original,
        normalized_query=normalized_query,
        did_you_mean=did_you_mean,
        price=price_constraint,
        tokens=corrected_tokens,
        semantic_query=effective_semantic_query,
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

