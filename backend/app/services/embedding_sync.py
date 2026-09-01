"""
Incremental Product Embedding Synchronization Service

Architecture:
-------------
1. PostgreSQL is the authoritative source of truth.
2. ChromaDB 'products' collection is the persistent HNSW vector index.
3. FastEmbed 'sentence-transformers/all-MiniLM-L6-v2' generates 384-dim dense vectors offline.
4. Deterministic SHA-256 content hashing on canonical product text detects changes.
5. Synchronization operations:
   - NEW: Product in PostgreSQL, missing in ChromaDB -> generate embedding & upsert.
   - MODIFIED: Product in both, hash mismatch -> regenerate embedding & upsert.
   - DELETED: Vector in ChromaDB, missing in PostgreSQL -> delete from ChromaDB.
   - UNCHANGED: Product in both, hash matches -> skip embedding generation.
   - LEGACY: Vector in ChromaDB without hash, text matches -> backfill metadata.
6. Thread-safe lock prevents concurrent sync executions.
"""

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from fastembed import TextEmbedding
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.product import Product

logger = logging.getLogger(__name__)

# Base Directories
SERVICES_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICES_DIR.parent
BACKEND_DIR = APP_DIR.parent

# Model & Collection Constants (must match semantic_search.py)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
COLLECTION_NAME = "products"
CHROMA_DIR = BACKEND_DIR / "data" / "chroma"
MODEL_DIR = BACKEND_DIR / "models" / "all-MiniLM-L6-v2"

# Concurrency lock
_sync_lock = threading.Lock()


@dataclass
class SyncResult:
    """Detailed summary report of an embedding synchronization run."""
    pg_count: int = 0
    chroma_count: int = 0
    new_count: int = 0
    updated_count: int = 0
    deleted_count: int = 0
    unchanged_count: int = 0
    failed_count: int = 0
    backfilled_hash_count: int = 0
    embedding_time_ms: float = 0.0
    total_time_ms: float = 0.0
    has_changes: bool = False
    errors: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [
            "=" * 60,
            "NORTHSTAR EMBEDDING SYNCHRONIZATION REPORT",
            "=" * 60,
            f"PostgreSQL Products:      {self.pg_count}",
            f"ChromaDB Initial Vectors: {self.chroma_count}",
            "-" * 60,
            f"New Embeddings Added:     {self.new_count}",
            f"Modified Embeddings:      {self.updated_count}",
            f"Stale Vectors Deleted:    {self.deleted_count}",
            f"Unchanged (Skipped):      {self.unchanged_count}",
            f"Legacy Hashes Backfilled: {self.backfilled_hash_count}",
            f"Failed Product Embeds:    {self.failed_count}",
            "-" * 60,
            f"Embedding Compute Time:   {self.embedding_time_ms:.1f} ms",
            f"Total Sync Duration:      {self.total_time_ms:.1f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)


def build_product_text(product: Product) -> str:
    """Build canonical searchable text for semantic embedding.
    
    Must remain identical across indexing, syncing, and testing.
    Format: Name | Brand: <brand> | Category: <category> | Tags: <tags> | <description>
    """
    parts = []

    if product.product_name:
        parts.append(product.product_name.strip())

    if product.brand:
        parts.append(f"Brand: {product.brand.strip()}")

    if product.category:
        parts.append(f"Category: {product.category.strip()}")

    if product.tags:
        if isinstance(product.tags, list):
            tags_str = ", ".join(str(tag).strip() for tag in product.tags if str(tag).strip())
        else:
            tags_str = str(product.tags).strip()
        if tags_str:
            parts.append(f"Tags: {tags_str}")

    if product.description:
        parts.append(product.description.strip())

    return " | ".join(parts)


def calculate_product_hash(product_text: str) -> str:
    """Generate deterministic SHA-256 hex digest of canonical product text."""
    return hashlib.sha256(product_text.encode("utf-8")).hexdigest()


def get_chroma_collection(create_if_missing: bool = True) -> Any:
    """Get or create the persistent ChromaDB 'products' collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if create_if_missing:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Offline product semantic embeddings",
                "embedding_model": MODEL_NAME,
                "embedding_dimension": EMBEDDING_DIMENSION,
            },
        )
    return client.get_collection(COLLECTION_NAME)


def is_sync_in_progress() -> bool:
    """Check if synchronization is currently running in this process."""
    return _sync_lock.locked()


def synchronize_embeddings(
    db: Session,
    batch_size: Optional[int] = None,
    force_rebuild: bool = False,
    non_blocking: bool = False,
) -> SyncResult:
    """Synchronize PostgreSQL product catalog with ChromaDB vector store.
    
    Args:
        db: Active SQLAlchemy database session.
        batch_size: Embedding batch size for FastEmbed (default from settings).
        force_rebuild: If True, re-embeds all products regardless of hash.
        non_blocking: If True, returns immediately if another sync is already in progress.
        
    Returns:
        SyncResult dataclass with comprehensive synchronization metrics.
    """
    effective_batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE or 64
    result = SyncResult()
    start_time = time.perf_counter()

    # 1. Acquire Concurrency Lock
    acquired = _sync_lock.acquire(blocking=not non_blocking)
    if not acquired:
        logger.warning("[embedding_sync] Synchronization already in progress. Skipping duplicate request.")
        result.errors.append("Sync already in progress")
        return result

    try:
        # 2. Connect to ChromaDB Collection
        collection = get_chroma_collection(create_if_missing=True)
        result.chroma_count = collection.count()

        # 3. Read All Products from PostgreSQL (Authoritative Source)
        products = db.execute(
            select(Product).order_by(Product.id.asc())
        ).scalars().all()
        result.pg_count = len(products)

        pg_product_map: Dict[str, Product] = {str(p.id): p for p in products}
        pg_hash_map: Dict[str, Tuple[str, str, Product]] = {}
        for p in products:
            p_text = build_product_text(p)
            p_hash = calculate_product_hash(p_text)
            pg_hash_map[str(p.id)] = (p_hash, p_text, p)

        # 4. Read Existing Vectors & Metadatas from ChromaDB
        chroma_ids: Set[str] = set()
        chroma_meta_map: Dict[str, Dict[str, Any]] = {}
        chroma_doc_map: Dict[str, str] = {}

        if result.chroma_count > 0:
            existing_data = collection.get(include=["metadatas", "documents"])
            raw_ids = existing_data.get("ids", [])
            raw_metas = existing_data.get("metadatas", []) or []
            raw_docs = existing_data.get("documents", []) or []

            chroma_ids = set(raw_ids)
            for i, vid in enumerate(raw_ids):
                chroma_meta_map[vid] = raw_metas[i] if (i < len(raw_metas) and raw_metas[i] is not None) else {}
                chroma_doc_map[vid] = raw_docs[i] if (i < len(raw_docs) and raw_docs[i] is not None) else ""

        # 5. Diff & Categorize Work
        # 5A. Stale Vectors to Delete (in ChromaDB, not in PostgreSQL)
        stale_ids = list(chroma_ids - set(pg_product_map.keys()))

        # 5B. Products to Embed (New or Modified)
        to_embed_new: List[Tuple[str, str, str, Product]] = []
        to_embed_updated: List[Tuple[str, str, str, Product]] = []
        to_backfill_metadata: List[Tuple[str, str, Product]] = []

        for pid, (p_hash, p_text, product) in pg_hash_map.items():
            if force_rebuild:
                to_embed_new.append((pid, p_hash, p_text, product))
                continue

            if pid not in chroma_ids:
                # Completely new product
                to_embed_new.append((pid, p_hash, p_text, product))
            else:
                meta = chroma_meta_map.get(pid, {})
                stored_hash = meta.get("embedding_hash")

                if stored_hash:
                    if stored_hash == p_hash:
                        # Unchanged product
                        result.unchanged_count += 1
                    else:
                        # Product text/metadata modified
                        to_embed_updated.append((pid, p_hash, p_text, product))
                else:
                    # Legacy vector without hash — check if document/metadata matches
                    stored_doc = chroma_doc_map.get(pid, "")
                    if stored_doc == p_text or (
                        meta.get("product_name") == product.product_name and
                        meta.get("brand") == product.brand and
                        meta.get("category") == product.category and
                        float(meta.get("price", 0)) == float(product.price or 0)
                    ):
                        # Vector already matches canonical text -> backfill hash metadata without re-embedding
                        to_backfill_metadata.append((pid, p_hash, product))
                    else:
                        # Text changed or cannot verify -> regenerate embedding
                        to_embed_updated.append((pid, p_hash, p_text, product))

        result.new_count = len(to_embed_new)
        result.updated_count = len(to_embed_updated)
        result.deleted_count = len(stale_ids)
        result.backfilled_hash_count = len(to_backfill_metadata)
        result.has_changes = bool(result.new_count or result.updated_count or result.deleted_count or result.backfilled_hash_count)

        # 6. Execute Vector Deletions
        if stale_ids:
            logger.info(f"[embedding_sync] Removing {len(stale_ids)} stale vectors from ChromaDB...")
            # Delete in chunks of 500
            for i in range(0, len(stale_ids), 500):
                del_chunk = stale_ids[i:i + 500]
                collection.delete(ids=del_chunk)

        # 7. Execute Metadata Backfills (no re-embedding)
        if to_backfill_metadata:
            logger.info(f"[embedding_sync] Backfilling embedding_hash for {len(to_backfill_metadata)} legacy vectors...")
            for i in range(0, len(to_backfill_metadata), 500):
                bf_chunk = to_backfill_metadata[i:i + 500]
                bf_ids = [item[0] for item in bf_chunk]
                bf_metas = []
                for item in bf_chunk:
                    p = item[2]
                    h = item[1]
                    bf_metas.append({
                        "product_id": int(p.id),
                        "product_name": str(p.product_name or ""),
                        "brand": str(p.brand or ""),
                        "category": str(p.category or ""),
                        "price": float(p.price or 0.0),
                        "embedding_hash": h,
                        "embedding_model": MODEL_NAME,
                    })
                collection.update(ids=bf_ids, metadatas=bf_metas)
                result.unchanged_count += len(bf_chunk)

        # 8. Generate & Upsert Embeddings for New + Modified Products
        items_to_embed = to_embed_new + to_embed_updated
        if items_to_embed:
            logger.info(
                f"[embedding_sync] Generating embeddings for {len(items_to_embed)} products "
                f"({len(to_embed_new)} new, {len(to_embed_updated)} updated) in batches of {effective_batch_size}..."
            )
            embed_t0 = time.perf_counter()

            # Load FastEmbed model with cached weights
            model = TextEmbedding(
                model_name=MODEL_NAME,
                cache_dir=str(MODEL_DIR),
            )

            for i in range(0, len(items_to_embed), effective_batch_size):
                batch = items_to_embed[i:i + effective_batch_size]
                b_ids = [item[0] for item in batch]
                b_hashes = [item[1] for item in batch]
                b_texts = [item[2] for item in batch]
                b_products = [item[3] for item in batch]

                try:
                    # Generate 384-dim dense vectors
                    b_vectors = list(model.embed(b_texts))
                    b_embeddings = [vec.tolist() for vec in b_vectors]

                    b_metadatas = []
                    for p, h in zip(b_products, b_hashes):
                        b_metadatas.append({
                            "product_id": int(p.id),
                            "product_name": str(p.product_name or ""),
                            "brand": str(p.brand or ""),
                            "category": str(p.category or ""),
                            "price": float(p.price or 0.0),
                            "embedding_hash": h,
                            "embedding_model": MODEL_NAME,
                        })

                    # Upsert into ChromaDB
                    collection.upsert(
                        ids=b_ids,
                        embeddings=b_embeddings,
                        documents=b_texts,
                        metadatas=b_metadatas,
                    )
                except Exception as e:
                    logger.error(f"[embedding_sync] Error embedding batch starting at index {i}: {e}")
                    # Attempt individual fallback for the batch to salvage valid products
                    for single_id, single_hash, single_text, single_prod in batch:
                        try:
                            s_vec = list(model.embed([single_text]))[0].tolist()
                            collection.upsert(
                                ids=[single_id],
                                embeddings=[s_vec],
                                documents=[single_text],
                                metadatas=[{
                                    "product_id": int(single_prod.id),
                                    "product_name": str(single_prod.product_name or ""),
                                    "brand": str(single_prod.brand or ""),
                                    "category": str(single_prod.category or ""),
                                    "price": float(single_prod.price or 0.0),
                                    "embedding_hash": single_hash,
                                    "embedding_model": MODEL_NAME,
                                }],
                            )
                        except Exception as single_err:
                            result.failed_count += 1
                            err_msg = f"Product ID {single_id} ({single_prod.product_name}): {single_err}"
                            result.errors.append(err_msg)
                            logger.error(f"[embedding_sync] {err_msg}")

            result.embedding_time_ms = (time.perf_counter() - embed_t0) * 1000

        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        # Invalidate LRU query embedding cache in semantic_search if any vectors changed
        if result.has_changes:
            try:
                from app.services.semantic_search import _get_query_embedding
                _get_query_embedding.cache_clear()
            except Exception:
                pass

        if result.has_changes:
            logger.info(
                f"[embedding_sync] Sync complete: +{result.new_count} new, "
                f"~{result.updated_count} updated, -{result.deleted_count} deleted, "
                f"={result.unchanged_count} unchanged, {result.failed_count} failed in {result.total_time_ms:.1f}ms"
            )
        else:
            logger.info(
                f"[embedding_sync] Sync complete: 0 changes detected ({result.unchanged_count} unchanged) in {result.total_time_ms:.1f}ms"
            )

        return result

    except Exception as e:
        logger.error(f"[embedding_sync] Fatal error during synchronization: {e}", exc_info=True)
        result.errors.append(str(e))
        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    finally:
        _sync_lock.release()
