import sys
from pathlib import Path

# Ensure backend root directory is in sys.path for app module imports
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.services.fuzzy_search import fuzzy_search_products


def run_test():
    test_queries = [
        "nik shose",
        "samsng phone",
        "wireles hedphone",
    ]

    # Accept CLI search query if provided by user
    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]

    session = SessionLocal()

    try:
        print("=" * 80)
        print("NORTHSTAR PRODUCT SEARCH — PG_TRGM FUZZY SEARCH TEST")
        print("=" * 80)

        for query in test_queries:
            print(f"\nQUERY: '{query}'")
            print("-" * 80)
            print(f"{'Fuzzy Score':<12} | {'Brand':<15} | {'Category':<15} | {'Product Name'}")
            print("-" * 80)

            results = fuzzy_search_products(session, query, limit=5, min_similarity=0.1)

            if not results:
                print("  (No products matched the fuzzy search threshold)")
                continue

            for res in results:
                p = res.product
                score_str = f"{res.fuzzy_score:.4f}"
                brand_str = (p.brand[:14] + "..") if len(p.brand) > 14 else p.brand
                cat_str = (p.category[:14] + "..") if len(p.category) > 14 else p.category
                name_str = (p.product_name[:38] + "..") if len(p.product_name) > 38 else p.product_name

                print(f"{score_str:<12} | {brand_str:<15} | {cat_str:<15} | {name_str}")

        print("\n" + "=" * 80)
        print("Fuzzy search test complete.")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Fuzzy search failed: {e}")
        print("Make sure you have run 'alembic upgrade head' to enable pg_trgm extension and indexes.")
    finally:
        session.close()


if __name__ == "__main__":
    run_test()
