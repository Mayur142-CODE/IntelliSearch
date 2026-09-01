"""
Fuzzy Search Service Module — Multi-Path Candidate Retrieval with Push-Down Filtering

Performance & Recall Design:
-----------------------------
Three-path candidate retrieval strategy balances speed with recall across
different query types (exact typos, moderate typos, prefix-of-word matches).
All paths are query-agnostic — no hardcoded product vocabulary.

Path 1 — Primary GIN-indexed pre-filter (default pg_trgm threshold ~0.3):
   Uses the `%` operator on product_name and brand GIN trigram indexes.
   Fastest path (~5-20ms). Handles well-formed queries and mild typos.

Path 2 — Lowered threshold GIN pre-filter (threshold 0.1):
   Activated when Path 1 returns 0 candidates. Temporarily lowers
   pg_trgm.similarity_threshold to 0.1 via SET LOCAL (transaction-scoped).

Path 3 — Word-similarity fallback (bounded sequential scan):
   Activated when Path 2 also returns 0 candidates. Uses explicit
   word_similarity() function. Bounded by LIMIT 200.

Push-Down Hard Constraints:
   Accepts min_price and max_price to filter candidates directly within SQL.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.product import Product

# Candidate limit for fallback paths to bound scoring cost
_FALLBACK_CANDIDATE_LIMIT = 200


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


def _safe(token: str) -> str:
    """Escape single quotes for safe SQL string embedding."""
    return token.replace("'", "''")


def _token_score_sql(token: str, alias: str = "p") -> str:
    """
    Build field-weighted trigram similarity SQL for one token using strict_word_similarity.
    Weights: product_name=50%, brand=20%, category=15%, tags=15%.
    Takes greatest(strict_word_similarity, similarity) for name/brand/category.
    """
    t = _safe(token)
    return (
        f"(GREATEST(strict_word_similarity('{t}', {alias}.product_name),"
        f" similarity('{t}', {alias}.product_name)) * 0.50"
        f" + GREATEST(strict_word_similarity('{t}', {alias}.brand),"
        f" similarity('{t}', {alias}.brand)) * 0.20"
        f" + GREATEST(strict_word_similarity('{t}', {alias}.category),"
        f" similarity('{t}', {alias}.category)) * 0.15"
        f" + strict_word_similarity('{t}', {alias}.tags) * 0.15)"
    )


def _fuzzy_score_sql(tokens: List[str], query: str, alias: str = "p") -> str:
    """
    Build complete multi-token fuzzy score SQL expression.
    - 1 token: token_score*0.75 + full_name_sim*0.25
    - N tokens: avg_tok*0.60 + avg_tok*coverage*0.25 + full_name_sim*0.15
    """
    n = len(tokens)
    safe_q = _safe(query)
    full_sim = (
        f"GREATEST(strict_word_similarity('{safe_q}', {alias}.product_name),"
        f" similarity('{safe_q}', {alias}.product_name))"
    )

    if n == 1:
        ts = _token_score_sql(tokens[0], alias)
        return f"({ts} * 0.75 + {full_sim} * 0.25)"

    tok_exprs = [_token_score_sql(t, alias) for t in tokens]
    max_tok = f"GREATEST({', '.join(tok_exprs)})"
    avg_tok = f"(({' + '.join(tok_exprs)}) / {n}.0)"
    coverage_cases = " + ".join(
        f"CASE WHEN ({te}) >= 0.15 THEN 1.0 ELSE 0.0 END" for te in tok_exprs
    )
    coverage = f"(({coverage_cases}) / {n}.0)"
    return (
        f"({max_tok} * 0.45"
        f" + {avg_tok} * 0.25"
        f" + {avg_tok} * {coverage} * 0.15"
        f" + {full_sim} * 0.15)"
    )


def _build_gin_conditions(tokens: List[str]) -> List[str]:
    """
    Build GIN-indexed % operator conditions for tokens >= 2 characters.
    Searches product_name and brand (both have GIN trigram indexes).

    NOTE: previously gated at >= 3 characters, which meant queries made
    entirely of short tokens (e.g. "hp", "tv", "3m") produced zero GIN
    conditions and short-circuited the whole fuzzy path to an empty
    result set (see fuzzy_search_products: `if not gin_conds: return []`).
    Lowered to >= 2 so short brand/model tokens still get fuzzy candidates.
    Single-character tokens are still excluded — trigram similarity is not
    meaningful below 2 characters.
    """
    conditions = []
    for t in tokens:
        if len(t) >= 2:
            st = _safe(t)
            conditions.append(f"product_name % '{st}'")
            conditions.append(f"brand % '{st}'")
    return conditions


def _build_ws_conditions(tokens: List[str], threshold: float = 0.35) -> List[str]:
    """
    Build strict_word_similarity conditions for fallback path.
    Searches product_name, brand, and category.

    Gate lowered from >= 3 to >= 2 characters to match _build_gin_conditions
    (see note there) so this fallback path can still fire for short-token
    queries instead of also returning nothing.
    """
    conditions = []
    for t in tokens:
        if len(t) >= 2:
            st = _safe(t)
            conditions.append(f"strict_word_similarity('{st}', product_name) >= {threshold}")
            conditions.append(f"strict_word_similarity('{st}', brand) >= {threshold}")
            conditions.append(f"strict_word_similarity('{st}', category) >= {threshold}")
    return conditions


def _build_scored_cte_sql(
    candidates_cte: str,
    score_sql: str,
    min_similarity: float,
    limit: int,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> str:
    """Build the full SQL with candidates CTE + scored CTE + final SELECT and push-down price filtering."""
    price_clauses = []
    if min_price is not None:
        price_clauses.append("price >= :min_price")
    if max_price is not None:
        price_clauses.append("price <= :max_price")

    price_where = f" AND {' AND '.join(price_clauses)}" if price_clauses else ""

    return f"""
        {candidates_cte},
        scored AS (
            SELECT
                p.id, p.product_name, p.description, p.brand,
                p.category, p.tags, p.price, p.image,
                {score_sql} AS fuzzy_score
            FROM products p
            JOIN candidates c ON p.id = c.id
        )
        SELECT id, product_name, description, brand, category,
               tags, price, image, fuzzy_score
        FROM scored
        WHERE fuzzy_score >= :min_sim{price_where}
        ORDER BY fuzzy_score DESC
        LIMIT :lim
    """


def _execute_and_fetch(db: Session, sql_str: str, params: Dict[str, Any]):
    """Execute a CTE query and return raw rows."""
    return db.execute(text(sql_str), params).fetchall()


def _build_results(db: Session, rows) -> List[FuzzySearchResult]:
    """Convert raw SQL rows into FuzzySearchResult objects via bulk ORM fetch."""
    if not rows:
        return []

    matched_ids = [row[0] for row in rows]
    id_to_score = {row[0]: float(row[8]) for row in rows}

    products = db.query(Product).filter(Product.id.in_(matched_ids)).all()
    id_to_product = {p.id: p for p in products}

    return [
        FuzzySearchResult(
            product=id_to_product[pid],
            fuzzy_score=id_to_score[pid],
        )
        for pid in matched_ids
        if pid in id_to_product
    ]


def fuzzy_search_products(
    db: Session,
    query: str,
    limit: int = 50,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_similarity: float = 0.01,
) -> List[FuzzySearchResult]:
    """
    Perform token-aware & field-weighted PostgreSQL pg_trgm fuzzy search
    using multi-path candidate retrieval with push-down price filtering.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    tokens = [t.lower() for t in cleaned_query.split() if t.strip()]
    if not tokens:
        return []

    score_sql = _fuzzy_score_sql(tokens, cleaned_query, alias="p")
    gin_conds = _build_gin_conditions(tokens)

    if not gin_conds:
        return []

    gin_where = " OR ".join(gin_conds)
    params: Dict[str, Any] = {"min_sim": min_similarity, "lim": limit}
    if min_price is not None:
        params["min_price"] = float(min_price)
    if max_price is not None:
        params["max_price"] = float(max_price)

    # ------------------------------------------------------------------
    # PATH 1: GIN % at default threshold (0.3)
    # ------------------------------------------------------------------
    candidates_cte = f"WITH candidates AS (SELECT DISTINCT id FROM products WHERE {gin_where})"
    sql_str = _build_scored_cte_sql(
        candidates_cte, score_sql, min_similarity, limit, min_price=min_price, max_price=max_price
    )
    rows = _execute_and_fetch(db, sql_str, params)

    if rows:
        return _build_results(db, rows)

    # ------------------------------------------------------------------
    # PATH 2: GIN % with lowered threshold (0.1)
    # ------------------------------------------------------------------
    candidates_cte_limited = (
        f"WITH candidates AS ("
        f"SELECT DISTINCT id FROM products WHERE {gin_where} LIMIT {_FALLBACK_CANDIDATE_LIMIT})"
    )
    sql_str = _build_scored_cte_sql(
        candidates_cte_limited, score_sql, min_similarity, limit, min_price=min_price, max_price=max_price
    )

    try:
        db.execute(text("SET LOCAL pg_trgm.similarity_threshold = 0.1"))
        rows = _execute_and_fetch(db, sql_str, params)
    finally:
        db.execute(text("RESET pg_trgm.similarity_threshold"))

    if rows:
        return _build_results(db, rows)

    # ------------------------------------------------------------------
    # PATH 3: word_similarity() fallback (bounded sequential scan)
    # ------------------------------------------------------------------
    ws_conds = _build_ws_conditions(tokens, threshold=0.35)
    if ws_conds:
        ws_where = " OR ".join(ws_conds)
        candidates_cte_ws = (
            f"WITH candidates AS ("
            f"SELECT DISTINCT id FROM products WHERE {ws_where} LIMIT {_FALLBACK_CANDIDATE_LIMIT})"
        )
        sql_str = _build_scored_cte_sql(
            candidates_cte_ws, score_sql, min_similarity, limit, min_price=min_price, max_price=max_price
        )
        rows = _execute_and_fetch(db, sql_str, params)

        if rows:
            return _build_results(db, rows)

    return []
