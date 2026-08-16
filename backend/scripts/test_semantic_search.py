"""
Local Semantic Search Service Test Script

Tests vector similarity search against the pre-computed product embeddings database.
"""

import sys
from pathlib import Path

# Ensure backend root directory is in sys.path for app imports
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.services.semantic_search import semantic_search_products


def run_test():
    test_queries = [
        "something to carry my laptop",
        "device for listening to music",
        "shoes for morning running",
        "something to charge my phone",
        "bag for traveling",
    ]

    # Accept CLI search query if provided by user
    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]

    session = SessionLocal()

    try:
        print("=" * 85)
        print("NORTHSTAR PRODUCT SEARCH — FASTEMBED LOCAL SEMANTIC SEARCH TEST")
        print("=" * 85)

        for query in test_queries:
            print(f"\nQUERY: '{query}'")
            print("-" * 85)
            print(f"{'Semantic Score':<15} | {'Brand':<15} | {'Category':<18} | {'Product Name'}")
            print("-" * 85)

            results = semantic_search_products(
                db=session,
                query=query,
                limit=5,
                min_similarity=0.0,
            )

            if not results:
                print("  (No products matched the semantic search threshold)")
                continue

            for res in results:
                p = res.product
                score_str = f"{res.semantic_score:.4f}"
                brand_str = (p.brand[:14] + "..") if len(p.brand) > 14 else p.brand
                cat_str = (p.category[:17] + "..") if len(p.category) > 17 else p.category
                name_str = (p.product_name[:35] + "..") if len(p.product_name) > 35 else p.product_name

                print(f"{score_str:<15} | {brand_str:<15} | {cat_str:<18} | {name_str}")

        print("\n" + "=" * 85)
        print("Semantic search test complete.")
        print("=" * 85)

    except Exception as e:
        print(f"\n[ERROR] Semantic search test failed: {e}")
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    run_test()
