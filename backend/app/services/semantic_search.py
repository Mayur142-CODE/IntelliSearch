"""
Semantic Search Service Module

Architecture & Optimization Design Rationale:
---------------------------------------------
1. Why embeddings are precomputed offline:
   - Generating 384-dimensional dense vector embeddings for thousands of products on every search
     request or during web server startup causes severe CPU spikes, memory bloat, and blocks request handlers.
   - Pre-computing vectors offline into NumPy binary arrays (.npy) allows instant server startup and
     near-zero disk I/O overhead during search operations.

2. Why the FastEmbed model is loaded locally:
   - Using the local cached ONNX model weights in `backend/models/all-MiniLM-L6-v2` enables fully
     offline execution without network dependencies, Hugging Face API rate limits, or external latency.

3. Why Cosine Similarity can be computed using Dot Product:
   - FastEmbed outputs L2-normalized unit vectors where ||v||_2 = 1.
   - Cosine Similarity formula: cos(θ) = (u · v) / (||u|| * ||v||).
   - When vectors u and v are unit-normalized, ||u|| = 1 and ||v|| = 1, reducing the formula to:
     cos(θ) = u · v (a single BLAS/LAPACK matrix-vector dot product `embeddings_matrix @ query_vec`).

4. Why only top product IDs are fetched from PostgreSQL:
   - Scanning 5,000+ database rows over network/IPC connections for every search request is slow and expensive.
   - NumPy computes vector similarities across all 4,999 products in ~1-2 milliseconds in memory.
   - We extract only the top-K product IDs (e.g. 20 IDs) and query PostgreSQL with `WHERE id IN (...)`,
     minimizing database load and network payload sizes while maintaining peak search throughput.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastembed import TextEmbedding
from sqlalchemy.orm import Session

from app.models.product import Product

# Base Directory Configurations
SERVICES_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICES_DIR.parent
BACKEND_DIR = APP_DIR.parent

# Semantic Search Constants
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Storage Paths
EMBEDDINGS_DIR = BACKEND_DIR / "data" / "embeddings"
MODEL_DIR = BACKEND_DIR / "models" / "all-MiniLM-L6-v2"

# Binary File Paths
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "product_embeddings.npy"
PRODUCT_IDS_FILE = EMBEDDINGS_DIR / "product_ids.npy"

# Module-Level Lazy Loaded Singleton Cache
_model: Optional[TextEmbedding] = None
_embeddings_matrix: Optional[np.ndarray] = None
_product_ids: Optional[np.ndarray] = None


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


def get_semantic_search_resources() -> Tuple[TextEmbedding, np.ndarray, np.ndarray]:
    """
    Module-level lazy-loaded singleton for FastEmbed model and NumPy embedding matrices.
    Loads files and model once per process into memory to ensure repeated searches do not hit disk.
    """
    global _model, _embeddings_matrix, _product_ids

    if _model is not None and _embeddings_matrix is not None and _product_ids is not None:
        return _model, _embeddings_matrix, _product_ids

    # 1. Error handling for missing binary files
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Product embeddings file missing at {EMBEDDINGS_FILE}. "
            "Please run 'python scripts/generate_embeddings.py' first."
        )

    if not PRODUCT_IDS_FILE.exists():
        raise FileNotFoundError(
            f"Product IDs file missing at {PRODUCT_IDS_FILE}. "
            "Please run 'python scripts/generate_embeddings.py' first."
        )

    # 2. Load pre-computed NumPy binary files into memory
    _embeddings_matrix = np.load(EMBEDDINGS_FILE)
    _product_ids = np.load(PRODUCT_IDS_FILE)

    # 3. Validate array shapes and non-emptiness
    if _embeddings_matrix.size == 0 or _product_ids.size == 0:
        raise ValueError("Loaded embedding matrix or product IDs array is empty.")

    if _embeddings_matrix.ndim != 2 or _embeddings_matrix.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding matrix shape (N, {EMBEDDING_DIMENSION}), "
            f"got {_embeddings_matrix.shape}."
        )

    if len(_embeddings_matrix) != len(_product_ids):
        raise ValueError(
            f"Mismatch between embeddings count ({len(_embeddings_matrix)}) "
            f"and product IDs count ({len(_product_ids)})."
        )

    # 4. Load local FastEmbed model using cached ONNX weights in MODEL_DIR
    _model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_DIR),
    )

    return _model, _embeddings_matrix, _product_ids


def semantic_search_products(
    db: Session,
    query: str,
    limit: int = 20,
    min_similarity: float = 0.0,
) -> List[SemanticSearchResult]:
    """
    Perform local semantic similarity search across product catalog.

    1. Embeds user query string into a 384-dimensional vector using local FastEmbed model.
    2. Computes matrix dot product (cosine similarity) against all stored product embeddings in NumPy.
    3. Ranks top-K product IDs by similarity score completely in-memory.
    4. Queries PostgreSQL ONLY for the top matching product IDs.
    5. Returns ranked list of SemanticSearchResult objects preserving exact score ordering.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    # Get cached model and numpy embedding matrices
    model, embeddings_matrix, product_ids = get_semantic_search_resources()

    # Embed query text into 384-dimensional vector
    query_vectors = list(model.embed([cleaned_query]))
    if not query_vectors:
        return []

    query_vec = np.array(query_vectors[0], dtype=np.float32)

    # Ensure query vector is unit normalized for accurate cosine similarity
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    # Fast matrix dot product for Cosine Similarity (u · v)
    similarity_scores = np.dot(embeddings_matrix, query_vec)

    # Filter indices matching min_similarity threshold
    if min_similarity > 0.0:
        valid_indices = np.where(similarity_scores >= min_similarity)[0]
        if valid_indices.size == 0:
            return []
        sorted_valid = valid_indices[np.argsort(similarity_scores[valid_indices])[::-1]]
        top_indices = sorted_valid[:limit]
    else:
        if len(similarity_scores) > limit:
            top_indices = np.argsort(similarity_scores)[::-1][:limit]
        else:
            top_indices = np.argsort(similarity_scores)[::-1]

    if len(top_indices) == 0:
        return []

    # Extract top product IDs and corresponding scores
    top_product_ids = [int(product_ids[idx]) for idx in top_indices]
    id_to_score = {int(product_ids[idx]): float(similarity_scores[idx]) for idx in top_indices}

    # Fetch ONLY the top matching Product records from PostgreSQL
    products = db.query(Product).filter(Product.id.in_(top_product_ids)).all()
    if not products:
        return []

    id_to_product = {p.id: p for p in products}

    # Construct results list preserving exact semantic ranking order
    results = []
    for p_id in top_product_ids:
        if p_id in id_to_product:
            results.append(
                SemanticSearchResult(
                    product=id_to_product[p_id],
                    semantic_score=id_to_score[p_id],
                )
            )

    return results
