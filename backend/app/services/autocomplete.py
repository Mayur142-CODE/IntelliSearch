"""
Dynamic Autocomplete Suggestion Engine

Generates query suggestions using the existing CatalogVocabulary singleton
loaded at startup. Zero hardcoded brands, categories, product names, or prices.

Suggestion sources:
1. Prefix matching against in-memory vocabulary indexes
2. Fuzzy/typo correction via CatalogVocabulary and JaroWinkler/Levenshtein matching
3. Multi-word completion (last-token matching with prefix prepending)
4. Dynamic price bucket suggestions from real PostgreSQL percentile data
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, distance
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.query_parser import CatalogVocabulary, _soundex, STOP_WORDS


# ============================================================================
# Suggestion Data Model
# ============================================================================

@dataclass
class Suggestion:
    """A single autocomplete suggestion."""
    text: str
    type: str          # brand | category | product | correction | price | phrase
    score: float = 0.0  # ranking score (higher = better)

    def to_dict(self) -> dict:
        return {"text": self.text, "type": self.type}


# ============================================================================
# Price Operator Detection (reuses operator word sets)
# ============================================================================

_UPPER_PRICE_OPERATORS = {
    "under", "unders", "undr", "below", "belw", "blo",
    "less", "lessthan", "upto", "max", "maximum", "within",
    "sub", "cheaper", "budget",
}

_LOWER_PRICE_OPERATORS = {
    "above", "abov", "abovee", "over", "ovr",
    "more", "morethan", "min", "minimum",
    "higher", "starting", "from", "least", "atleast",
}

_RANGE_PRICE_OPERATORS = {
    "between", "btwn", "range",
}

_ALL_PRICE_OPERATORS = _UPPER_PRICE_OPERATORS | _LOWER_PRICE_OPERATORS | _RANGE_PRICE_OPERATORS


def _consonant_skeleton(word: str) -> str:
    """Extract consonant skeleton preserving first character and stripping vowels."""
    w = word.lower().strip()
    if not w:
        return ""
    first = w[0]
    rest = re.sub(r'[aeiouy]', '', w[1:])
    return first + rest


def _fuzzy_matches_price_operator(word: str) -> Optional[str]:
    """Check if a word fuzzy-matches any price operator.

    Returns the operator category ('upper', 'lower', 'range') or None.
    """
    w = word.lower().strip(".,;:!?")
    if not w or len(w) < 2:
        return None

    # Exact membership
    if w in _UPPER_PRICE_OPERATORS:
        return "upper"
    if w in _LOWER_PRICE_OPERATORS:
        return "lower"
    if w in _RANGE_PRICE_OPERATORS:
        return "range"

    # Fuzzy match (handles typos like "unders", "abov", "belw")
    for op in _UPPER_PRICE_OPERATORS:
        if distance.Levenshtein.distance(w, op) <= 1 or fuzz.ratio(w, op) >= 80:
            return "upper"
    for op in _LOWER_PRICE_OPERATORS:
        if distance.Levenshtein.distance(w, op) <= 1 or fuzz.ratio(w, op) >= 80:
            return "lower"
    for op in _RANGE_PRICE_OPERATORS:
        if distance.Levenshtein.distance(w, op) <= 1 or fuzz.ratio(w, op) >= 80:
            return "range"

    return None


# ============================================================================
# Dynamic Price Bucket Generation
# ============================================================================

def _compute_price_buckets(
    db: Session,
    semantic_prefix: str,
    operator_type: str,
) -> List[float]:
    """Compute dynamic price buckets from actual product prices in PostgreSQL.

    Uses percentile aggregation on products matching the semantic prefix.
    Returns 4-6 rounded price points.
    """
    prefix_tokens = [t.strip() for t in semantic_prefix.split() if t.strip() and len(t.strip()) >= 2]

    params: Dict = {}
    if not prefix_tokens:
        where_clause = "1=1"
    else:
        conditions = []
        for i, token in enumerate(prefix_tokens):
            param_key = f"tok_{i}"
            conditions.append(
                f"(product_name ILIKE :{param_key} OR brand ILIKE :{param_key} "
                f"OR category ILIKE :{param_key} OR tags ILIKE :{param_key} "
                f"OR description ILIKE :{param_key})"
            )
            params[param_key] = f"%{token}%"
        where_clause = " AND ".join(conditions)

    # Query percentile price points
    sql = text(f"""
        SELECT
            COUNT(*) as cnt,
            MIN(price) as min_p,
            MAX(price) as max_p,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY price) as p10,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) as p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price) as p50,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) as p75,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY price) as p90
        FROM products
        WHERE {where_clause}
    """)

    try:
        row = db.execute(sql, params).fetchone()
    except Exception:
        # Fallback to global query if the conditional query failed
        try:
            sql_fallback = text("""
                SELECT
                    COUNT(*) as cnt,
                    MIN(price) as min_p,
                    MAX(price) as max_p,
                    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY price) as p10,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) as p25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price) as p50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) as p75,
                    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY price) as p90
                FROM products
            """)
            row = db.execute(sql_fallback).fetchone()
        except Exception:
            return []

    if not row or not row.cnt or row.cnt == 0:
        # If no products matched the prefix, fallback to global distribution
        try:
            sql_fallback = text("""
                SELECT
                    COUNT(*) as cnt,
                    MIN(price) as min_p,
                    MAX(price) as max_p,
                    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY price) as p10,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) as p25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price) as p50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) as p75,
                    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY price) as p90
                FROM products
            """)
            row = db.execute(sql_fallback).fetchone()
        except Exception:
            return []

    if not row or not row.cnt or row.cnt == 0:
        return []

    # Collect raw percentile values
    raw_points = []
    for val in [row.p10, row.p25, row.p50, row.p75, row.p90]:
        if val is not None:
            raw_points.append(float(val))

    if not raw_points:
        return []

    # Round to human-friendly numbers
    rounded = sorted(set(_round_price(p) for p in raw_points if p > 0))
    if not rounded:
        return []

    # Filter out duplicates that are too close (within 15% of each other)
    filtered = [rounded[0]]
    for p in rounded[1:]:
        if p > filtered[-1] * 1.15:
            filtered.append(p)

    return filtered[:5]


def _round_price(price: float) -> float:
    """Round a price to a human-friendly number."""
    if price <= 0:
        return 0
    if price < 100:
        return round(price / 10) * 10
    if price < 500:
        return round(price / 50) * 50
    if price < 2000:
        return round(price / 100) * 100
    if price < 10000:
        return round(price / 500) * 500
    return round(price / 1000) * 1000


# ============================================================================
# Core Autocomplete Engine
# ============================================================================

def generate_suggestions(
    db: Session,
    query: str,
    max_results: int = 8,
) -> List[Suggestion]:
    """Generate autocomplete suggestions for a partial query.

    Uses the CatalogVocabulary singleton (already loaded at startup)
    plus dynamic PostgreSQL queries for price buckets and keyword follow-ons.
    """
    q = query.strip()
    if not q:
        return []

    vocab = CatalogVocabulary.get_instance()
    vocab.load(db)

    tokens = q.split()
    last_token = tokens[-1].lower() if tokens else ""
    prefix_tokens = tokens[:-1] if len(tokens) > 1 else []
    prefix_str = " ".join(prefix_tokens)

    suggestions: List[Suggestion] = []

    # ----------------------------------------------------------------
    # 1. Check if last token is an incomplete price operator
    # ----------------------------------------------------------------
    price_op_type = _fuzzy_matches_price_operator(last_token)

    # Also check two-word operators like "less than", "more than", "up to"
    if not price_op_type and len(tokens) >= 2:
        two_word = f"{tokens[-2].lower()} {last_token}"
        if two_word in ("less than", "more than", "up to", "at least",
                        "at most", "higher than", "cheaper than",
                        "starting from"):
            if two_word in ("less than", "up to", "at most", "cheaper than"):
                price_op_type = "upper"
            else:
                price_op_type = "lower"
            prefix_tokens = tokens[:-2]
            prefix_str = " ".join(prefix_tokens)

    if price_op_type:
        semantic_prefix = prefix_str if prefix_str else q.rsplit(last_token, 1)[0].strip()
        corrected_prefix = _correct_prefix_tokens(vocab, semantic_prefix)

        buckets = _compute_price_buckets(db, corrected_prefix, price_op_type)
        op_word = " ".join(tokens[len(prefix_tokens):])

        for i, price_val in enumerate(buckets):
            price_display = _format_price(price_val)
            suggestion_text = f"{semantic_prefix} {op_word} {price_display}".strip()
            suggestions.append(Suggestion(
                text=suggestion_text,
                type="price",
                score=0.99 - (i * 0.01),
            ))

        # Also add non-price suggestions for the semantic prefix
        if semantic_prefix:
            prefix_suggestions = _generate_term_suggestions(vocab, semantic_prefix, max_results=3, db=db)
            suggestions.extend(prefix_suggestions)

        return _deduplicate_and_rank(suggestions, max_results)

    # ----------------------------------------------------------------
    # 2. Check if user is typing a partial price number after operator
    #    e.g. "phone under 2" → suggest "phone under 200", "phone under 2000"
    # ----------------------------------------------------------------
    if len(tokens) >= 2:
        potential_num = last_token.rstrip("k").rstrip("K")
        prev_word = tokens[-2].lower()
        prev_is_price_op = _fuzzy_matches_price_operator(prev_word)

        if prev_is_price_op and potential_num.replace(".", "").isdigit():
            semantic_prefix = " ".join(tokens[:-2])
            corrected_prefix = _correct_prefix_tokens(vocab, semantic_prefix)
            buckets = _compute_price_buckets(db, corrected_prefix, prev_is_price_op)

            typed_val = float(potential_num) if potential_num else 0
            for i, price_val in enumerate(buckets):
                price_str = _format_price(price_val)
                if str(int(price_val)).startswith(str(int(typed_val))) and price_val != typed_val:
                    suggestion_text = f"{semantic_prefix} {prev_word} {price_str}".strip()
                    suggestions.append(Suggestion(
                        text=suggestion_text,
                        type="price",
                        score=0.95 - (i * 0.01),
                    ))

            exact_text = f"{semantic_prefix} {prev_word} {last_token}".strip()
            suggestions.insert(0, Suggestion(text=exact_text, type="price", score=0.99))

            return _deduplicate_and_rank(suggestions, max_results)

    # ----------------------------------------------------------------
    # 3. Term-based suggestions (prefix + fuzzy + correction + SQL keywords)
    # ----------------------------------------------------------------
    suggestions = _generate_term_suggestions(vocab, q, max_results=max_results, db=db)

    return _deduplicate_and_rank(suggestions, max_results)


def _generate_term_suggestions(
    vocab: CatalogVocabulary,
    query: str,
    max_results: int = 8,
    db: Session = None,
) -> List[Suggestion]:
    """Generate term-based suggestions (prefix, fuzzy, correction, SQL keyword)."""
    q = query.strip().lower()
    if not q:
        return []

    tokens = q.split()
    last_token = tokens[-1] if tokens else ""
    prefix_tokens = tokens[:-1] if len(tokens) > 1 else []
    prefix_str = " ".join(prefix_tokens)
    is_single = len(tokens) == 1

    suggestions: List[Suggestion] = []
    seen_texts: set = set()

    def _add(text: str, stype: str, score: float):
        normalized = text.strip().lower()
        if normalized and normalized not in seen_texts:
            seen_texts.add(normalized)
            suggestions.append(Suggestion(text=text.strip(), type=stype, score=score))

    # --- A. Direct brand / category detection for whole query & last token ---
    brand_match = vocab.find_matching_brand(q if is_single else last_token, is_single_token_query=True)
    if brand_match:
        brand_name, b_score, is_exact = brand_match
        sug_text = f"{prefix_str} {brand_name}".strip() if prefix_str else brand_name
        _add(sug_text, "brand", 0.98 if is_exact else 0.94)

    cat_match = vocab.find_matching_category(q if is_single else last_token)
    if cat_match:
        cat_name, c_score = cat_match
        sug_text = f"{prefix_str} {cat_name}".strip() if prefix_str else cat_name
        _add(sug_text, "category", 0.95)

    # Brand / Category prefix match
    for brand_lower, canonical in vocab.brand_lower_map.items():
        if brand_lower.startswith(q):
            _add(canonical, "brand", 1.0)
        elif last_token and brand_lower.startswith(last_token) and len(last_token) >= 2:
            sug = f"{prefix_str} {canonical}".strip() if prefix_str else canonical
            _add(sug, "brand", 0.92)

    for cat_lower, canonical in vocab.category_lower_map.items():
        if cat_lower.startswith(q):
            _add(canonical, "category", 0.96)
        elif last_token and cat_lower.startswith(last_token) and len(last_token) >= 2:
            sug = f"{prefix_str} {canonical}".strip() if prefix_str else canonical
            _add(sug, "category", 0.90)

    # --- B. Vocabulary token prefix matching ---
    all_vocabs = [
        (vocab.brand_vocab, "brand"),
        (vocab.category_vocab, "category"),
        (vocab.product_name_vocab, "product"),
        (vocab.tag_vocab, "product"),
    ]

    for field_vocab, stype in all_vocabs:
        for entry in field_vocab:
            tok = entry.token
            if len(tok) < 2:
                continue

            if tok.startswith(last_token) and tok != last_token and len(last_token) >= 2:
                display = entry.canonical if hasattr(entry, 'canonical') and stype in ("brand", "category") else tok
                suggestion_text = f"{prefix_str} {display}".strip() if prefix_str else display
                freq_score = math.log1p(entry.count) / math.log1p(vocab.max_frequency)
                _add(suggestion_text, stype, 0.75 + 0.15 * freq_score)

    # --- C. Fuzzy matching against vocabulary (JaroWinkler & RapidFuzz) ---
    if len(last_token) >= 3 and last_token not in STOP_WORDS:
        # First check standard query parser correct_token
        corrections = vocab.correct_token(last_token)
        for corr in corrections:
            if corr.source_field != "uncorrected" and corr.confidence >= 0.5:
                stype = "brand" if corr.source_field == "brand" else "category" if corr.source_field == "category" else "correction"
                suggestion_text = f"{prefix_str} {corr.corrected}".strip() if prefix_str else corr.corrected
                _add(suggestion_text, stype, corr.confidence * 0.90)

        # Also search vocabulary entries with JaroWinkler and consonant skeleton for broader typo recovery (e.g. spik->speaker, lpt->laptop, nyk->nike)
        tok_soundex = _soundex(last_token)
        tok_skel = _consonant_skeleton(last_token)
        for field_vocab, stype in all_vocabs:
            for entry in field_vocab:
                if len(entry.token) < 3:
                    continue

                r_score = float(fuzz.ratio(last_token, entry.token))
                jw_score = float(distance.JaroWinkler.similarity(last_token, entry.token)) * 100.0
                dist = distance.Levenshtein.distance(last_token, entry.token)
                max_sim = max(r_score, jw_score)
                phonetic = entry.soundex == tok_soundex

                entry_skel = _consonant_skeleton(entry.token)
                skel_dist = distance.Levenshtein.distance(tok_skel, entry_skel)
                skel_match = (skel_dist <= 1 and len(tok_skel) >= 2) or (tok_skel == entry_skel)

                if (dist <= 2 and max_sim >= 65.0) or (phonetic and dist <= 3) or (skel_match and max_sim >= 60.0) or (max_sim >= 75.0 and dist <= 3):
                    display = entry.canonical if hasattr(entry, 'canonical') and stype in ("brand", "category") else entry.token
                    suggestion_text = f"{prefix_str} {display}".strip() if prefix_str else display
                    freq_score = math.log1p(entry.count) / math.log1p(vocab.max_frequency)
                    field_bonus = 0.12 if stype == "brand" else 0.08 if stype == "category" else 0.05
                    skel_bonus = 0.10 if skel_match else 0.0
                    phonetic_bonus = 0.08 if phonetic else 0.0
                    score = 0.50 * (max_sim / 100.0) + 0.25 * freq_score + field_bonus + skel_bonus + phonetic_bonus
                    _add(suggestion_text, stype, score)

    # --- D. SQL keyword extraction (multi-word completions from product names) ---
    if db is not None and len(q) >= 3:
        _add_sql_keyword_suggestions(db, q, _add)

    # --- E. Full query as-is (if not already present) ---
    _add(q, "phrase", 0.4)

    return suggestions


def _add_sql_keyword_suggestions(db: Session, query: str, _add_fn):
    """Extract common keyword completions from product names matching the query."""
    try:
        query_tokens = [t for t in query.split() if len(t) >= 2]
        if not query_tokens:
            return

        conditions = []
        params = {}
        for i, tok in enumerate(query_tokens):
            param_key = f"kw_{i}"
            conditions.append(f"LOWER(product_name) LIKE :{param_key}")
            params[param_key] = f"%{tok.lower()}%"

        where = " AND ".join(conditions)
        sql = text(f"""
            SELECT LOWER(product_name) as pname
            FROM products
            WHERE {where}
            LIMIT 150
        """)

        rows = db.execute(sql, params).fetchall()
        if not rows:
            return

        from collections import Counter
        phrase_counter = Counter()
        query_lower = query.lower()

        for row in rows:
            pname = row.pname
            words = re.findall(r'[a-z0-9]+', pname)

            for i, w in enumerate(words):
                if w.startswith(query_tokens[-1]) or query_tokens[-1].startswith(w):
                    for length in range(2, 4):
                        end = i + length
                        if end <= len(words):
                            fragment = " ".join(words[i:end])
                            if len(fragment) > len(query_lower) and fragment != query_lower:
                                phrase_counter[fragment] += 1

        for phrase, count in phrase_counter.most_common(5):
            if count >= 2:
                if len(query_tokens) > 1:
                    earlier = " ".join(query_tokens[:-1])
                    suggestion = f"{earlier} {phrase}"
                else:
                    suggestion = phrase
                _add_fn(suggestion, "product", 0.78 + min(count / 50.0, 0.12))

    except Exception:
        pass


def _correct_prefix_tokens(vocab: CatalogVocabulary, prefix: str) -> str:
    """Apply typo correction to prefix tokens for better SQL matching."""
    if not prefix:
        return prefix

    tokens = prefix.split()
    corrected = []
    for tok in tokens:
        if len(tok) >= 3 and tok.lower() not in STOP_WORDS:
            corrections = vocab.correct_token(tok)
            if corrections and corrections[0].source_field != "uncorrected":
                corrected.append(corrections[0].corrected)
            else:
                # Also check brand match
                bm = vocab.find_matching_brand(tok, is_single_token_query=True)
                if bm:
                    corrected.append(bm[0])
                else:
                    corrected.append(tok)
        else:
            corrected.append(tok)
    return " ".join(corrected)


def _format_price(price: float) -> str:
    """Format a price value for display in suggestions."""
    if price >= 1000 and price % 1000 == 0:
        return f"{int(price // 1000)}k"
    return str(int(price))


def _deduplicate_and_rank(
    suggestions: List[Suggestion],
    max_results: int,
) -> List[Suggestion]:
    """Deduplicate suggestions by normalized text and return top-ranked."""
    seen: set = set()
    unique: List[Suggestion] = []
    for s in suggestions:
        key = s.text.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    unique.sort(key=lambda s: s.score, reverse=True)
    return unique[:max_results]
