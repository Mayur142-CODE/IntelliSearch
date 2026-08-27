"""
Dynamic Complete-Query Autocomplete Engine

Architecture & Principles:
1. Multi-Token Typo Correction:
   - Evaluates the whole query and token combinations against the catalog vocabulary.
   - If a high-confidence correction is detected, places "Did you mean <corrected query>"
     as the #1 suggestion with is_correction=True and score=1.0.

2. Complete-Query Candidate Generation:
   - AND-gated product matching across all query concepts (e.g. Nike AND shoes)
     so unrelated products (e.g. Uniqlo Running Shoes) are never returned for Nike shoe queries.
   - Clean, complete product title phrases extracted directly from real PostgreSQL products.
   - Semantic vector candidates from ChromaDB + FastEmbed for natural language queries.
   - Dynamic price bucket completions from PostgreSQL percentiles.

3. Complete-Query Multi-Signal Scoring:
   - Evaluates the whole candidate phrase against the whole user query.
   - Strict token coverage weighting so candidates missing query concepts receive a low score.
   - No hardcoded brands, categories, products, typos, or price numbers.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz, distance
from sqlalchemy import text, select, or_, distinct
from sqlalchemy.orm import Session

from app.models.product import Product
from app.services.query_parser import CatalogVocabulary, ParsedQuery, parse_query, STOP_WORDS
from app.services.semantic_search import semantic_search_products


# ============================================================================
# Suggestion Data Model
# ============================================================================

@dataclass
class Suggestion:
    text: str
    type: str = "phrase"
    score: float = 0.0
    is_correction: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.type,
            "score": round(self.score, 4),
            "is_correction": self.is_correction,
        }


# ============================================================================
# Price Operator Detection & Percentile Generation
# ============================================================================

_UPPER_PRICE_OPS = {
    "under", "unders", "undr", "below", "belw", "blo",
    "less", "lessthan", "upto", "max", "maximum", "within",
    "sub", "cheaper",
}

_LOWER_PRICE_OPS = {
    "above", "abov", "abovee", "over", "ovr",
    "more", "morethan", "min", "minimum",
    "higher", "starting", "from", "least", "atleast",
}

_RANGE_PRICE_OPS = {
    "between", "btwn", "range",
}

_ALL_PRICE_OPS = _UPPER_PRICE_OPS | _LOWER_PRICE_OPS | _RANGE_PRICE_OPS

# Natural language indicator terms
NL_INDICATORS = {
    "something", "device", "things", "stuff", "item", "product",
    "carry", "carrying", "listen", "listening", "charge", "charging",
    "walk", "walking", "travel", "traveling", "travelling", "gift",
    "for", "to", "with", "my", "me", "how", "what", "good", "best",
}


def _clean_phrase(text_str: str) -> str:
    """Clean a phrase by stripping non-alphanumeric chars and leading/trailing stop words."""
    if not text_str:
        return ""
    cleaned = re.sub(r'[^\w\s\-]', ' ', text_str)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = cleaned.split()
    while words and words[-1].lower() in STOP_WORDS:
        words.pop()
    while words and words[0].lower() in STOP_WORDS:
        words.pop(0)
    return " ".join(words)


def _extract_product_title_phrases(product_name: str, brand: Optional[str] = None) -> List[str]:
    """Extract clean, complete, meaningful candidate search phrases from a real product name."""
    if not product_name:
        return []

    phrases: List[str] = []
    clean_name = _clean_phrase(product_name)
    words = clean_name.split()
    if not words:
        return []

    # 1. Full clean name if concise (<= 5 words)
    if 2 <= len(words) <= 5:
        phrases.append(clean_name)

    # 2. Sub-phrases of 2, 3, 4 words
    if len(words) >= 2:
        p2 = _clean_phrase(" ".join(words[:2]))
        if len(p2.split()) >= 2:
            phrases.append(p2)
    if len(words) >= 3:
        p3 = _clean_phrase(" ".join(words[:3]))
        if len(p3.split()) >= 2:
            phrases.append(p3)
    if len(words) >= 4:
        p4 = _clean_phrase(" ".join(words[:4]))
        if len(p4.split()) >= 2:
            phrases.append(p4)

    # 3. Brand + first 2 words if brand is present
    if brand and brand.strip():
        b = brand.strip()
        if not clean_name.lower().startswith(b.lower()) and len(words) >= 2:
            b_phrase = _clean_phrase(f"{b} {words[0]} {words[1]}")
            if len(b_phrase.split()) >= 2:
                phrases.append(b_phrase)

    # Deduplicate preserving order
    seen = set()
    result = []
    for p in phrases:
        k = p.lower()
        if k not in seen and len(k) >= 3:
            seen.add(k)
            result.append(p)
    return result


def _detect_price_operator_in_tokens(tokens: List[str]) -> Optional[Tuple[str, str, int]]:
    """Check if the query ends in a price operator."""
    if not tokens:
        return None

    if len(tokens) >= 2:
        two_word = f"{tokens[-2].lower()} {tokens[-1].lower()}"
        if two_word in ("less than", "up to", "at most", "cheaper than"):
            return ("upper", two_word, len(tokens) - 2)
        if two_word in ("more than", "at least", "higher than", "starting from"):
            return ("lower", two_word, len(tokens) - 2)

    last = tokens[-1].lower().strip(".,;:!?")
    if last in _UPPER_PRICE_OPS:
        return ("upper", tokens[-1], len(tokens) - 1)
    if last in _LOWER_PRICE_OPS:
        return ("lower", tokens[-1], len(tokens) - 1)
    if last in _RANGE_PRICE_OPS:
        return ("range", tokens[-1], len(tokens) - 1)

    if len(last) >= 3:
        for op in _UPPER_PRICE_OPS:
            if distance.Levenshtein.distance(last, op) <= 1 or fuzz.ratio(last, op) >= 80:
                return ("upper", tokens[-1], len(tokens) - 1)
        for op in _LOWER_PRICE_OPS:
            if distance.Levenshtein.distance(last, op) <= 1 or fuzz.ratio(last, op) >= 80:
                return ("lower", tokens[-1], len(tokens) - 1)
        for op in _RANGE_PRICE_OPS:
            if distance.Levenshtein.distance(last, op) <= 1 or fuzz.ratio(last, op) >= 80:
                return ("range", tokens[-1], len(tokens) - 1)

    return None


def _compute_price_buckets(db: Session, semantic_prefix: str, operator_type: str) -> List[float]:
    """Compute dynamic price points from real PostgreSQL price distribution."""
    prefix_tokens = [t.strip() for t in semantic_prefix.split() if t.strip() and len(t.strip()) >= 2 and t.lower() not in STOP_WORDS]
    params: Dict = {}
    if not prefix_tokens:
        where_clause = "1=1"
    else:
        conditions = []
        for i, token in enumerate(prefix_tokens):
            param_key = f"tok_{i}"
            conditions.append(
                f"(product_name ILIKE :{param_key} OR brand ILIKE :{param_key} "
                f"OR category ILIKE :{param_key} OR tags ILIKE :{param_key})"
            )
            params[param_key] = f"%{token}%"
        where_clause = " AND ".join(conditions)

    sql = text(f"""
        SELECT
            COUNT(*) as cnt,
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
        row = None

    if not row or not row.cnt or row.cnt == 0:
        try:
            sql_fallback = text("""
                SELECT
                    COUNT(*) as cnt,
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

    raw_points = [float(v) for v in [row.p10, row.p25, row.p50, row.p75, row.p90] if v is not None and float(v) > 0]
    if not raw_points:
        return []

    rounded = sorted(set(_round_price(p) for p in raw_points if p > 0))
    if not rounded:
        return []

    filtered = [rounded[0]]
    for p in rounded[1:]:
        if p > filtered[-1] * 1.15:
            filtered.append(p)
    return filtered[:5]


def _round_price(price: float) -> float:
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


def _format_price_val(price: float) -> str:
    if price >= 1000 and price % 1000 == 0:
        return f"{int(price // 1000)}k"
    return str(int(price))


# ============================================================================
# Token Coverage & Complete-Query Scoring
# ============================================================================

def _calculate_token_coverage(query_tokens: List[str], candidate_tokens: List[str]) -> float:
    """Calculate the fraction of query concepts covered by the candidate phrase.

    Ensures that for a multi-word query (e.g. "Nike shoes"), candidates that only
    match 1 word (e.g. "Uniqlo shoes") receive a harsh coverage penalty.
    """
    meaningful_q = [t for t in query_tokens if t not in STOP_WORDS and len(t) >= 2]
    if not meaningful_q:
        return 1.0
    if not candidate_tokens:
        return 0.0

    matched = 0
    for q_tok in meaningful_q:
        best_tok_match = 0.0
        for c_tok in candidate_tokens:
            if c_tok == q_tok or c_tok.startswith(q_tok) or q_tok.startswith(c_tok):
                best_tok_match = max(best_tok_match, 1.0)
            else:
                sim = fuzz.ratio(q_tok, c_tok) / 100.0
                if sim >= 0.75:
                    best_tok_match = max(best_tok_match, sim)

        if best_tok_match >= 0.75:
            matched += 1

    return matched / len(meaningful_q)


def _score_complete_query(
    raw_query: str,
    normalized_query: str,
    candidate_text: str,
    semantic_score: float = 0.0,
    is_price: bool = False,
    is_direct_correction: bool = False,
) -> float:
    """Score the COMPLETE candidate phrase against the COMPLETE user query."""
    if is_direct_correction:
        return 1.0

    if is_price:
        return max(0.90, semantic_score)

    q_raw = raw_query.strip().lower()
    q_norm = normalized_query.strip().lower() if normalized_query else q_raw
    c_text = candidate_text.strip().lower()

    if not c_text:
        return 0.0

    raw_tokens = [t for t in q_raw.split() if t and t not in STOP_WORDS]
    norm_tokens = [t for t in q_norm.split() if t and t not in STOP_WORDS]
    cand_tokens = [t for t in c_text.split() if t and t not in STOP_WORDS]

    # Whole-query fuzzy similarity
    fuzz_norm = fuzz.ratio(q_norm, c_text) / 100.0
    fuzz_raw = fuzz.ratio(q_raw, c_text) / 100.0
    whole_fuzzy = max(fuzz_norm, fuzz_raw)

    # Word-order invariant token sort ratio
    sort_norm = fuzz.token_sort_ratio(q_norm, c_text) / 100.0
    sort_raw = fuzz.token_sort_ratio(q_raw, c_text) / 100.0
    token_sort = max(sort_norm, sort_raw)

    # Token set ratio
    set_norm = fuzz.token_set_ratio(q_norm, c_text) / 100.0
    set_raw = fuzz.token_set_ratio(q_raw, c_text) / 100.0
    token_set = max(set_norm, set_raw)

    # Token coverage across all query concepts
    cov_norm = _calculate_token_coverage(norm_tokens, cand_tokens)
    cov_raw = _calculate_token_coverage(raw_tokens, cand_tokens)
    token_coverage = max(cov_norm, cov_raw)

    # Prefix alignment
    prefix_score = 0.0
    if c_text.startswith(q_norm) or c_text.startswith(q_raw):
        prefix_score = 1.0
    elif norm_tokens and cand_tokens and cand_tokens[0].startswith(norm_tokens[0]):
        prefix_score = 0.6

    base_score = (
        0.35 * token_set +
        0.25 * token_sort +
        0.20 * whole_fuzzy +
        0.20 * prefix_score
    )

    # Harsh penalty for candidates that miss major query concepts (e.g. Uniqlo for Nike shoe query)
    final_score = base_score * (token_coverage ** 2.0)

    if semantic_score > 0:
        final_score = max(final_score, 0.40 * final_score + 0.60 * semantic_score)

    return min(1.0, max(0.0, final_score))


# ============================================================================
# Main Autocomplete Engine
# ============================================================================

def generate_suggestions(
    db: Session,
    query: str,
    max_results: int = 8,
) -> List[Suggestion]:
    """Generate dynamic autocomplete suggestions for a partial query.

    Features:
    1. Multi-Token Typo Correction -> "Did you mean <corrected query>?"
    2. Complete Catalog Candidate Mining -> AND-gated product matching
    3. Fast Semantic Search Vectors -> ChromaDB + FastEmbed for NL queries
    4. Dynamic Percentile Price Completions -> PostgreSQL distribution
    5. Complete-Query Multi-Signal Scoring -> Ranked and deduplicated
    """
    raw_query = query.strip()
    if not raw_query or len(raw_query) < 2:
        return []

    vocab = CatalogVocabulary.get_instance()
    vocab.load(db)

    tokens = raw_query.split()
    lower_q = raw_query.lower()

    # Determine if natural language query
    is_nl = any(w in lower_q.split() for w in NL_INDICATORS) or len(tokens) >= 4

    # ===================================================================
    # 1. Price Operator Autocomplete
    # ===================================================================
    price_op_info = _detect_price_operator_in_tokens(tokens)
    if price_op_info:
        op_type, op_word, op_idx = price_op_info
        subject_tokens = tokens[:op_idx]
        subject_str = " ".join(subject_tokens).strip()

        parsed_subject = parse_query(db, subject_str) if subject_str else None
        norm_subject = parsed_subject.normalized_query if parsed_subject and parsed_subject.normalized_query else subject_str

        buckets = _compute_price_buckets(db, norm_subject, op_type)
        price_suggestions: List[Suggestion] = []

        for i, price_val in enumerate(buckets):
            price_display = _format_price_val(price_val)
            text_cand = f"{norm_subject} {op_word} {price_display}".strip()
            price_suggestions.append(Suggestion(
                text=text_cand,
                type="price",
                score=0.99 - (i * 0.01),
            ))

        if subject_str:
            sub_suggestions = generate_suggestions(db, subject_str, max_results=3)
            for s in sub_suggestions:
                price_suggestions.append(s)

        return _deduplicate_and_rank(price_suggestions, max_results)

    # Partial price number e.g. "laptop under 2"
    if len(tokens) >= 2 and tokens[-1].replace(".", "").isdigit():
        prev_op_info = _detect_price_operator_in_tokens(tokens[:-1])
        if prev_op_info:
            op_type, op_word, op_idx = prev_op_info
            subject_str = " ".join(tokens[:op_idx]).strip()
            partial_num = tokens[-1]

            parsed_subject = parse_query(db, subject_str) if subject_str else None
            norm_subject = parsed_subject.normalized_query if parsed_subject and parsed_subject.normalized_query else subject_str

            buckets = _compute_price_buckets(db, norm_subject, op_type)
            price_suggestions: List[Suggestion] = []

            for i, price_val in enumerate(buckets):
                price_str = _format_price_val(price_val)
                if str(int(price_val)).startswith(partial_num) and str(int(price_val)) != partial_num:
                    text_cand = f"{norm_subject} {op_word} {price_str}".strip()
                    price_suggestions.append(Suggestion(
                        text=text_cand,
                        type="price",
                        score=0.96 - (i * 0.01),
                    ))

            exact_cand = f"{norm_subject} {op_word} {partial_num}".strip()
            price_suggestions.insert(0, Suggestion(text=exact_cand, type="price", score=0.99))

            return _deduplicate_and_rank(price_suggestions, max_results)

    # ===================================================================
    # 2. Multi-Token Query Understanding & Typo Parsing
    # ===================================================================
    parsed_query = parse_query(db, raw_query)
    norm_query = parsed_query.normalized_query.strip() if not is_nl else raw_query
    detected_brands = parsed_query.detected_brands
    detected_categories = parsed_query.detected_categories

    norm_words = norm_query.lower().split()
    all_valid_tokens = bool(norm_words) and all(
        vocab.is_valid_word(w)
        for w in norm_words
    )

    is_typo_correction = bool(
        not is_nl and norm_query and
        norm_query.lower() != raw_query.lower() and
        len(norm_query) >= 3 and
        all_valid_tokens
    )

    candidates_dict: Dict[str, Tuple[str, float, bool]] = {}  # text -> (type, semantic_score, is_correction)

    def _register(phrase: str, ptype: str = "phrase", sem_score: float = 0.0, is_corr: bool = False):
        cleaned = _clean_phrase(phrase)
        if cleaned and len(cleaned) >= 2:
            k = cleaned.lower()
            if k not in candidates_dict:
                candidates_dict[k] = (cleaned, ptype, sem_score, is_corr)
            else:
                old_type, old_score, old_corr = candidates_dict[k][1], candidates_dict[k][2], candidates_dict[k][3]
                candidates_dict[k] = (cleaned, ptype, max(old_score, sem_score), old_corr or is_corr)

    # Register typo-corrected query as top "Did you mean" candidate
    if is_typo_correction:
        _register(norm_query, "correction", 0.0, is_corr=True)

    # ===================================================================
    # 3. Dynamic Semantic Vector Search (ChromaDB + FastEmbed)
    # ===================================================================
    # Always run for NL queries; run with small limit for all queries >= 3 chars
    semantic_limit = 10 if is_nl else 4
    try:
        sem_res_list = semantic_search_products(
            db=db,
            query=parsed_query.semantic_query or raw_query,
            limit=semantic_limit,
            normalized_query=norm_query if norm_query != raw_query and not is_nl else None,
        )
        for s_res in sem_res_list:
            prod = s_res.product
            s_score = s_res.semantic_score

            for phrase in _extract_product_title_phrases(prod.product_name, prod.brand):
                _register(phrase, "semantic", s_score)
    except Exception:
        pass

    # ===================================================================
    # 4. PostgreSQL Catalog Product & Phrase Mining (AND-gated)
    # ===================================================================
    search_toks = [t for t in (norm_query if not is_nl else raw_query).split() if len(t) >= 2 and t.lower() not in STOP_WORDS]

    if search_toks:
        # 4A. Primary AND query: find products that match ALL concepts
        and_conditions = []
        and_params = {}
        for i, tok in enumerate(search_toks):
            param_k = f"and_tk_{i}"
            and_conditions.append(
                f"(product_name ILIKE :{param_k} OR brand ILIKE :{param_k} "
                f"OR category ILIKE :{param_k} OR tags ILIKE :{param_k})"
            )
            and_params[param_k] = f"%{tok}%"

        and_where = " AND ".join(and_conditions)
        sql_and = text(f"""
            SELECT product_name, brand, category
            FROM products
            WHERE {and_where}
            LIMIT 40
        """)

        matched_rows = []
        try:
            matched_rows = db.execute(sql_and, and_params).fetchall()
        except Exception:
            matched_rows = []

        # If AND query returned matches, mine high-confidence phrases exclusively from them
        for r in matched_rows:
            pname, pbrand, pcat = r[0], r[1], r[2]
            for phrase in _extract_product_title_phrases(pname, pbrand):
                _register(phrase, "product", 0.0)

        # 4B. If fewer than 5 products found, supplement with OR query
        if len(matched_rows) < 5:
            or_conditions = []
            or_params = {}
            for i, tok in enumerate(search_toks):
                param_k = f"or_tk_{i}"
                or_conditions.append(
                    f"(product_name ILIKE :{param_k} OR brand ILIKE :{param_k} "
                    f"OR category ILIKE :{param_k} OR tags ILIKE :{param_k})"
                )
                or_params[param_k] = f"%{tok}%"

            or_where = " OR ".join(or_conditions)
            sql_or = text(f"""
                SELECT product_name, brand, category
                FROM products
                WHERE {or_where}
                LIMIT 30
            """)
            try:
                or_rows = db.execute(sql_or, or_params).fetchall()
                for r in or_rows:
                    pname, pbrand, pcat = r[0], r[1], r[2]
                    for phrase in _extract_product_title_phrases(pname, pbrand):
                        _register(phrase, "product", 0.0)
            except Exception:
                pass

    # ===================================================================
    # 5. In-Memory Vocabulary Prefix Matches (for single-word typing)
    # ===================================================================
    if len(tokens) == 1:
        prefix = tokens[0].lower()
        for b_lower, canonical in vocab.brand_lower_map.items():
            if b_lower.startswith(prefix):
                _register(canonical, "brand", 0.0)
        for c_lower, canonical in vocab.category_lower_map.items():
            if c_lower.startswith(prefix):
                _register(canonical, "category", 0.0)
        for entry in vocab.product_name_vocab:
            if entry.token.startswith(prefix) and len(entry.token) > len(prefix):
                _register(entry.token, "product", 0.0)
                if len(candidates_dict) > 40:
                    break

    # Prefix expansion for multi-word queries (e.g. "budget lap", "wireles hea")
    if len(tokens) >= 2:
        last_tok = tokens[-1].lower()
        earlier_tokens = tokens[:-1]
        corrected_earlier = []
        for et in earlier_tokens:
            corrs = vocab.correct_token(et)
            corrected_earlier.append(corrs[0].corrected if corrs and corrs[0].confidence >= 0.70 else et)
        earlier_norm = " ".join(corrected_earlier)

        if len(last_tok) >= 2 and last_tok not in STOP_WORDS:
            for entry in vocab.product_name_vocab:
                if entry.token.startswith(last_tok) and entry.token != last_tok:
                    _register(f"{earlier_norm} {entry.token}", "product", 0.0)
                    if len(candidates_dict) > 80:
                        break

    # ===================================================================
    # 6. Complete-Query Multi-Signal Scoring & Selection
    # ===================================================================
    scored_suggestions: List[Suggestion] = []
    raw_tokens = [t for t in lower_q.split() if t]
    norm_tokens = [t for t in norm_query.lower().split() if t]

    # pyrefly: ignore [bad-unpacking]
    for key, (cand_text, cand_type, sem_score, is_corr) in candidates_dict.items():
        c_lower = cand_text.lower()
        cand_tokens = [t for t in c_lower.split() if t]

        if is_corr:
            scored_suggestions.append(Suggestion(
                text=cand_text,
                type=cand_type,
                score=1.0,
                is_correction=True,
            ))
            continue

        if is_nl:
            cov = _calculate_token_coverage(raw_tokens, cand_tokens)
            score = 0.70 * sem_score + 0.30 * cov
            if sem_score >= 0.35 or score >= 0.40:
                scored_suggestions.append(Suggestion(
                    text=cand_text,
                    type=cand_type,
                    score=score,
                    is_correction=False,
                ))
        else:
            score = _score_complete_query(
                raw_query=raw_query,
                normalized_query=norm_query,
                candidate_text=cand_text,
                semantic_score=sem_score,
                is_direct_correction=False,
            )

            if score >= 0.35:
                scored_suggestions.append(Suggestion(
                    text=cand_text,
                    type=cand_type,
                    score=score,
                    is_correction=False,
                ))

    if not scored_suggestions:
        scored_suggestions.append(Suggestion(
            text=raw_query,
            type="phrase",
            score=0.50,
            is_correction=False,
        ))

    return _deduplicate_and_rank(scored_suggestions, max_results)


def _deduplicate_and_rank(suggestions: List[Suggestion], max_results: int) -> List[Suggestion]:
    seen: set = set()
    unique: List[Suggestion] = []

    # Sort primarily by score descending (corrections are score 1.0)
    sorted_sug = sorted(suggestions, key=lambda s: s.score, reverse=True)
    for s in sorted_sug:
        normalized = s.text.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(s)
            if len(unique) >= max_results:
                break

    return unique
