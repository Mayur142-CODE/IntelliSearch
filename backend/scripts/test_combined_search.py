"""
Combined Search Ranking Engine Test Script

Tests candidate union across 4 candidate sources (Exact, Partial, Fuzzy, Semantic) and multi-signal ranking.
"""

import sys
from pathlib import Path

# Ensure backend root directory is in sys.path for app imports
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.services.search_ranking import search_products


def run_test():
    test_suite = [
        ("EXACT QUERIES", [
            "nike",
            "samsung",
            "laptop",
            "footwear",
        ]),
        ("PREFIX / PARTIAL QUERIES", [
            "foot",
            "footwe",
            "lapt",
            "head",
            "wire",
            "phon",
            "sams",
            "run",
        ]),
        ("TYPO / FUZZY QUERIES", [
            "lptop",
            "botle",
            "nik shose",
            "samsng phone",
            "wireles hedphone",
        ]),
        ("SEMANTIC QUERIES", [
            "something to carry my laptop",
            "device for listening to music",
            "shoes for morning running",
            "something to charge my phone",
            "bag for traveling",
        ]),
        ("BRAND QUERIES", [
            "anker",
        ]),
        ("CATEGORY QUERIES", [
            "electronics",
            "fashion",
            "audio",
        ]),
        ("MULTI-WORD QUERIES", [
            "wireless headphones",
            "gaming laptop",
            "running shoes",
            "phone case",
        ]),
        ("NO RESULT QUERY", [
            "nonexistentproduct12345xyz",
        ]),
    ]


    session = SessionLocal()

    try:
        print("=" * 115)
        print("NORTHSTAR PRODUCT SEARCH — 4-SOURCE HYBRID CANDIDATE & RANKING TEST")
        print("=" * 115)

        for category_title, queries in test_suite:
            print(f"\n>>> {category_title}")

            for query in queries:
                print(f"\nQUERY: '{query}'")
                print("-" * 115)
                print(
                    f"{'FINAL':<7} | {'EXACT':<6} | {'PARTIAL':<7} | {'FUZZY':<6} | {'SEMANTIC':<8} | "
                    f"{'BRAND':<14} | {'CATEGORY':<16} | {'PRODUCT NAME'}"
                )
                print("-" * 115)

                results = search_products(
                    db=session,
                    query=query,
                    limit=5,
                    candidate_limit=50,
                )

                if not results:
                    print("  (No products matched the search query)")
                    continue

                for r in results:
                    p = r.product
                    f_score = f"{r.final_score:.3f}"
                    e_score = f"{r.exact_score:.2f}"
                    p_score = f"{r.partial_score:.2f}"
                    fz_score = f"{r.fuzzy_score:.2f}"
                    s_score = f"{r.semantic_score:.2f}"

                    brand_str = (p.brand[:13] + "..") if len(p.brand) > 13 else p.brand
                    cat_str = (p.category[:15] + "..") if len(p.category) > 15 else p.category
                    name_str = (p.product_name[:32] + "..") if len(p.product_name) > 32 else p.product_name

                    print(
                        f"{f_score:<7} | {e_score:<6} | {p_score:<7} | {fz_score:<6} | {s_score:<8} | "
                        f"{brand_str:<14} | {cat_str:<16} | {name_str}"
                    )

        print("\n" + "=" * 115)
        print("4-Source hybrid search ranking test complete.")
        print("=" * 115)

    except Exception as e:
        print(f"\n[ERROR] Combined search test failed: {e}")
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    run_test()
