"""
Offline Product Embeddings Generation Script

This script:
1. Fetches all products from PostgreSQL in batches.
2. Constructs a rich textual representation for each product (combining product_name, brand, category, tags, and description).
3. Uses FastEmbed (all-MiniLM-L6-v2) to compute normalized 384-dim vector embeddings.
4. Saves product embeddings (float32) and product IDs (int64) into numpy binary files:
   - backend/data/embeddings/product_embeddings.npy (shape: N, 384)
   - backend/data/embeddings/product_ids.npy        (shape: N,)
5. Ensures exact 1:1 index alignment between product IDs and vector embedding rows.
"""

import sys
import time
from pathlib import Path

# Ensure backend root directory is in sys.path for app imports
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
from fastembed import TextEmbedding
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.product import Product
from app.services.semantic_search import (
    EMBEDDING_DIMENSION,
    EMBEDDINGS_DIR,
    EMBEDDINGS_FILE,
    MODEL_DIR,
    MODEL_NAME,
    PRODUCT_IDS_FILE,
)

BATCH_SIZE = 256


def build_product_text(product: Product) -> str:
    """Build a rich, structured textual representation of a product for semantic embedding."""
    parts = []

    if product.product_name:
        parts.append(product.product_name)

    if product.brand:
        parts.append(f"Brand: {product.brand}")

    if product.category:
        parts.append(f"Category: {product.category}")

    if product.tags:
        if isinstance(product.tags, list):
            parts.append(f"Tags: {', '.join(product.tags)}")
        else:
            parts.append(f"Tags: {product.tags}")

    if product.description:
        parts.append(product.description)

    return " | ".join(parts)


def generate_embeddings():
    """Batch load products, compute embeddings using FastEmbed, and save to NumPy .npy files."""
    start_time = time.time()

    # Ensure output directories exist
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NORTHSTAR PRODUCT SEARCH — LOCAL EMBEDDING GENERATION")
    print("=" * 80)
    print(f"Model:               {MODEL_NAME}")
    print(f"Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"Model Cache Dir:     {MODEL_DIR}")
    print(f"Embeddings Out Dir:  {EMBEDDINGS_DIR}")
    print("-" * 80)

    # Initialize local FastEmbed model
    print("\nLoading FastEmbed model...")
    model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_DIR),
    )
    print("Model successfully loaded.\n")

    db: Session = SessionLocal()

    try:
        # Fetch total product count
        total_products = db.query(Product).count()
        if total_products == 0:
            print("[WARNING] No products found in the database.")
            return

        print(f"Processing {total_products} products in batches of {BATCH_SIZE}...")

        all_product_ids = []
        all_embeddings = []
        processed_count = 0

        # Query products in ordered batches by ID for reproducible, stable indexing
        offset = 0
        while offset < total_products:
            batch_products = (
                db.query(Product)
                .order_by(Product.id.asc())
                .offset(offset)
                .limit(BATCH_SIZE)
                .all()
            )

            if not batch_products:
                break

            batch_ids = [p.id for p in batch_products]
            batch_texts = [build_product_text(p) for p in batch_products]

            # Generate embeddings for current batch using FastEmbed
            batch_vectors = list(model.embed(batch_texts))

            all_product_ids.extend(batch_ids)
            all_embeddings.extend(batch_vectors)

            processed_count += len(batch_products)
            offset += BATCH_SIZE

            progress_pct = (processed_count / total_products) * 100
            print(f"  Processed {processed_count}/{total_products} products ({progress_pct:.1f}%)...")

        # Convert lists to NumPy arrays with specified memory-efficient dtypes
        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        product_ids_array = np.array(all_product_ids, dtype=np.int64)

        # Save to disk as .npy binary files (overwrites existing files cleanly)
        np.save(EMBEDDINGS_FILE, embeddings_matrix)
        np.save(PRODUCT_IDS_FILE, product_ids_array)

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total Products Processed: {processed_count}")
        print(f"Embeddings Matrix Shape:  {embeddings_matrix.shape}")
        print(f"Product IDs Array Shape: {product_ids_array.shape}")
        print(f"Saved Embeddings File:   {EMBEDDINGS_FILE}")
        print(f"Saved Product IDs File:  {PRODUCT_IDS_FILE}")
        print(f"Total Generation Time:   {elapsed_time:.2f} seconds")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Failed to generate embeddings: {e}")
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    generate_embeddings()
