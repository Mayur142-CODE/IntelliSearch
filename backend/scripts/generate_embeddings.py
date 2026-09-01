"""
Offline Product Embeddings Generation using ChromaDB.

This script:
1. Fetches products from PostgreSQL in batches.
2. Builds rich product text.
3. Generates 384-dimensional embeddings using local FastEmbed.
4. Stores embeddings directly in persistent ChromaDB.
5. Stores product metadata alongside each vector.

No .npy files are created.
"""

import sys
import time
from pathlib import Path

# Ensure backend root is available for imports
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import chromadb
from fastembed import TextEmbedding
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.product import Product
from app.services.embedding_sync import build_product_text, calculate_product_hash


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_DIMENSION = 384

BATCH_SIZE = 256

CHROMA_DIR = BACKEND_DIR / "data" / "chroma"

COLLECTION_NAME = "products"


# ---------------------------------------------------------
# Main Generation
# ---------------------------------------------------------

def generate_embeddings():

    start_time = time.time()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NORTHSTAR PRODUCT SEARCH — CHROMADB EMBEDDING GENERATION")
    print("=" * 80)

    print(f"Model:              {MODEL_NAME}")
    print(f"Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"ChromaDB Directory: {CHROMA_DIR}")
    print(f"Batch Size:         {BATCH_SIZE}")
    print("-" * 80)

    # -----------------------------------------------------
    # Load local embedding model
    # -----------------------------------------------------

    print("\nLoading local FastEmbed model...")

    model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(BACKEND_DIR / "models"),
    )

    print("Model successfully loaded.")

    # -----------------------------------------------------
    # Initialize ChromaDB
    # -----------------------------------------------------

    print("\nInitializing ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Delete old collection if it exists.
    # This guarantees a completely fresh embedding index.
    try:
        chroma_client.delete_collection(
            name=COLLECTION_NAME
        )

        print(f"Deleted existing collection: {COLLECTION_NAME}")

    except Exception:
        print("No existing collection found.")

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Offline product semantic embeddings",
            "embedding_model": MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
        },
    )

    print(f"Created collection: {COLLECTION_NAME}")

    # -----------------------------------------------------
    # Database
    # -----------------------------------------------------

    db: Session = SessionLocal()

    try:

        total_products = db.query(Product).count()

        if total_products == 0:
            print("\n[WARNING] No products found in PostgreSQL.")
            return

        print(
            f"\nProcessing {total_products} products "
            f"in batches of {BATCH_SIZE}..."
        )

        processed_count = 0

        offset = 0

        # -------------------------------------------------
        # Process products
        # -------------------------------------------------

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

            batch_ids = [
                str(product.id)
                for product in batch_products
            ]

            batch_texts = [
                build_product_text(product)
                for product in batch_products
            ]

            # Generate embeddings
            batch_vectors = list(
                model.embed(batch_texts)
            )

            # Convert embeddings to normal Python lists
            batch_embeddings = [
                vector.tolist()
                for vector in batch_vectors
            ]

            # Metadata
            batch_metadata = []

            for product, text in zip(batch_products, batch_texts):

                metadata = {
                    "product_id": int(product.id),
                    "product_name": str(
                        product.product_name or ""
                    ),
                    "brand": str(
                        product.brand or ""
                    ),
                    "category": str(
                        product.category or ""
                    ),
                    "price": float(
                        product.price or 0
                    ),
                    "embedding_hash": calculate_product_hash(text),
                    "embedding_model": MODEL_NAME,
                }

                batch_metadata.append(metadata)

            # -------------------------------------------------
            # Store in ChromaDB
            # -------------------------------------------------

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadata,
            )

            processed_count += len(batch_products)

            offset += BATCH_SIZE

            progress = (
                processed_count / total_products
            ) * 100

            print(
                f"  Processed "
                f"{processed_count}/{total_products} "
                f"({progress:.1f}%)"
            )

        # -------------------------------------------------
        # Verification
        # -------------------------------------------------

        total_vectors = collection.count()

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("CHROMADB EMBEDDING GENERATION COMPLETE")
        print("=" * 80)

        print(
            f"Products Processed:  {processed_count}"
        )

        print(
            f"Vectors in ChromaDB: {total_vectors}"
        )

        print(
            f"Embedding Dimension: {EMBEDDING_DIMENSION}"
        )

        print(
            f"Generation Time:     {elapsed_time:.2f}s"
        )

        print(
            f"ChromaDB Location:   {CHROMA_DIR}"
        )

        print("=" * 80)

        # -------------------------------------------------
        # Safety verification
        # -------------------------------------------------

        if total_vectors != processed_count:

            raise RuntimeError(
                "CRITICAL: Product count and ChromaDB "
                "vector count do not match."
            )

        print("\nSUCCESS:")
        print(
            "Every product has a corresponding "
            "embedding in ChromaDB."
        )

    except Exception as e:

        print(
            f"\n[ERROR] Embedding generation failed: {e}"
        )

        raise

    finally:

        db.close()


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    generate_embeddings()