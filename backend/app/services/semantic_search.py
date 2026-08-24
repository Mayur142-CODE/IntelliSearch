"""
Semantic Search Service Module — ChromaDB Persistent Vector Engine

Architecture & Optimization Design:
------------------------------------
1. Persistent ChromaDB Vector Store:
   - Connects to ChromaDB PersistentClient at backend/data/chroma.
   - Vector collection 'products' contains pre-computed 384-dim embeddings.
   - Supports metadata filtering (e.g. price range pushed directly into HNSW index).

2. Distance to Similarity Metric Conversion:
   - ChromaDB computes squared Euclidean distance (L2^2) for normalized unit vectors:
     ||u - v||^2 = ||u||^2 + ||v||^2 - 2(u · v) = 2 - 2*cos(θ)
   - Cosine Similarity = 1.0 - (distance / 2.0), bounded in [0.0, 1.0].

3. Local FastEmbed ONNX Model:
   - sentence-transformers/all-MiniLM-L6-v2 runs locally with zero external network calls.
   - LRU query embedding cache (128 entries) eliminates redundant ONNX inference.

4. Multi-Variant Query Embedding for Typo Generalization:
   - When typo-normalized query variants are generated, queries all vector candidates
     and merges best cosine similarity scores per product into the semantic candidate pool.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from fastembed import TextEmbedding
import numpy as np
from sqlalchemy.orm import Session

from app.models.product import Product

# Base Directory Configurations
SERVICES_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICES_DIR.parent
BACKEND_DIR = APP_DIR.parent

# Semantic Search Constants
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
COLLECTION_NAME = "products"

# Storage Paths
CHROMA_DIR = BACKEND_DIR / "data" / "chroma"
MODEL_DIR = BACKEND_DIR / "models" / "all-MiniLM-L6-v2"

# Module-Level Lazy Loaded Singleton Cache
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Any] = None
_model: Optional[TextEmbedding] = None


@dataclass
class SemanticSearchResult:
    """Dataclass holding a matched Product instance along with its semantic similarity score."""
    product: Product
    semantic_score: float

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
            "semantic_score": round(self.semantic_score, 4),
        }


def get_semantic_search_resources() -> Tuple[TextEmbedding, Any]:
    """
    Module-level lazy-loaded singleton for FastEmbed model and ChromaDB collection.
    Initializes ChromaDB PersistentClient and loads ONNX embedding weights once per process.
    """
    global _chroma_client, _collection, _model

    if _model is not None and _collection is not None:
        return _model, _collection

    # 1. Initialize Persistent ChromaDB Client
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory missing at {CHROMA_DIR}. "
            "Please run 'python scripts/generate_embeddings.py' first."
        )

    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        _collection = _chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in ChromaDB at {CHROMA_DIR}: {e}"
        )

    # 2. Load local FastEmbed model using cached ONNX weights in MODEL_DIR
    _model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_DIR),
    )

    return _model, _collection


@lru_cache(maxsize=128)
def _get_query_embedding(query: str) -> np.ndarray:
    """
    LRU-cached query embedding to avoid redundant ONNX inference for repeated queries.
    Bounded at 128 entries. Returns a unit-normalized float32 query vector.
    """
    model, _ = get_semantic_search_resources()
    query_vectors = list(model.embed([query]))
    if not query_vectors:
        raise ValueError(f"FastEmbed returned no vectors for query: {query!r}")
    query_vec = np.array(query_vectors[0], dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm
    return query_vec


def _build_chroma_where(
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build ChromaDB where metadata filter."""
    clauses = []

    if min_price is not None:
        clauses.append({"price": {"$gte": float(min_price)}})
    if max_price is not None:
        clauses.append({"price": {"$lte": float(max_price)}})
    if brand is not None:
        clauses.append({"brand": {"$eq": str(brand)}})
    if category is not None:
        clauses.append({"category": {"$eq": str(category)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def semantic_search_products(
    db: Session,
    query: str,
    limit: int = 50,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    brand_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    min_similarity: float = 0.0,
    normalized_query: Optional[str] = None,
    normalized_queries: Optional[List[str]] = None,
) -> List[SemanticSearchResult]:
    """
    Perform local semantic similarity search across product catalog using ChromaDB.

    1. Embeds user query string into a 384-dimensional vector using local FastEmbed model.
    2. If normalized query variants are provided, searches all variants and merges best scores.
    3. Queries ChromaDB persistent HNSW index with push-down metadata price/brand filters.
    4. Converts returned L2 distances into true cosine similarity scores.
    5. Queries PostgreSQL ONLY for the top matching product IDs.
    6. Returns ranked list of SemanticSearchResult objects preserving exact score ordering.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    # Get cached model and Chroma collection
    model, collection = get_semantic_search_resources()

    # Collect query vectors to search
    query_vectors_to_search: List[List[float]] = []
    try:
        query_vec = _get_query_embedding(cleaned_query)
        query_vectors_to_search.append(query_vec.tolist())
    except Exception:
        pass

    all_normalized = list(normalized_queries) if normalized_queries else []
    if normalized_query and normalized_query not in all_normalized:
        all_normalized.append(normalized_query)

    for nq in all_normalized:
        if nq and nq.strip() and nq.strip().lower() != cleaned_query.lower():
            try:
                norm_vec = _get_query_embedding(nq.strip())
                query_vectors_to_search.append(norm_vec.tolist())
            except Exception:
                pass

    if not query_vectors_to_search:
        return []

    # Push down metadata filters to ChromaDB vector search
    where_clause = _build_chroma_where(
        min_price=min_price,
        max_price=max_price,
        brand=brand_filter,
        category=category_filter,
    )

    id_to_score: Dict[int, float] = {}

    for q_vec_list in query_vectors_to_search:
        try:
            query_kwargs = {
                "query_embeddings": [q_vec_list],
                "n_results": min(limit, collection.count() or limit),
                "include": ["metadatas", "distances"],
            }
            if where_clause:
                query_kwargs["where"] = where_clause

            query_res = collection.query(**query_kwargs)

            if query_res and query_res.get("ids") and query_res["ids"][0]:
                raw_ids = query_res["ids"][0]
                raw_distances = query_res["distances"][0]
                for str_id, dist in zip(raw_ids, raw_distances):
                    pid = int(str_id)
                    sim_score = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                    if sim_score >= min_similarity:
                        id_to_score[pid] = max(id_to_score.get(pid, 0.0), sim_score)
        except Exception:
            continue

    if not id_to_score:
        return []

    # Sort product IDs by best semantic similarity score
    sorted_pids = sorted(id_to_score.keys(), key=lambda pid: id_to_score[pid], reverse=True)[:limit]

    # Fetch matching Product records from PostgreSQL
    products = db.query(Product).filter(Product.id.in_(sorted_pids)).all()
    if not products:
        return []

    id_to_product = {p.id: p for p in products}

    # Construct results list preserving exact semantic ranking order
    results: List[SemanticSearchResult] = []
    for p_id in sorted_pids:
        if p_id in id_to_product:
            results.append(
                SemanticSearchResult(
                    product=id_to_product[p_id],
                    semantic_score=id_to_score[p_id],
                )
            )

    return results
