import sys
from pathlib import Path

# Ensure backend root directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_hybrid_search_api():
    test_queries = [
        "nike",
        "samsung",
        "laptop",
        "head",
        "run",
        "wire",
        "phone",
        "nik shose",
        "samsng phone",
        "wireles hedphone",
        "something to carry my laptop",
        "device for listening to music",
        "shoes for morning running",
        "something to charge my phone",
        "bag for traveling",
        "nonexistentproduct12345xyz",
    ]

    print("=" * 115)
    print("NORTHSTAR PRODUCT SEARCH — FASTAPI HYBRID SEARCH API TEST")
    print("=" * 115)

    for q in test_queries:
        response = client.get("/search", params={"q": q, "limit": 5})
        assert response.status_code == 200, f"Endpoint failed for '{q}': {response.status_code}"
        data = response.json()

        assert "query" in data
        assert "count" in data
        assert "results" in data
        assert data["count"] == len(data["results"])

        print(f"\nQUERY: '{data['query']}' | COUNT: {data['count']}")
        if data["results"]:
            print(
                f"{'FINAL':<7} | {'EXACT':<6} | {'PARTIAL':<7} | {'FUZZY':<6} | {'SEMANTIC':<8} | "
                f"{'BRAND':<14} | {'CATEGORY':<16} | {'PRODUCT NAME'}"
            )
            print("-" * 115)
            for r in data["results"]:
                # Verify all score keys exist in API payload
                assert "exact_score" in r
                assert "partial_score" in r
                assert "fuzzy_score" in r
                assert "semantic_score" in r
                assert "final_score" in r

                f_score = f"{r['final_score']:.3f}"
                e_score = f"{r['exact_score']:.2f}"
                p_score = f"{r['partial_score']:.2f}"
                fz_score = f"{r['fuzzy_score']:.2f}"
                s_score = f"{r['semantic_score']:.2f}"

                brand_str = (r['brand'][:13] + "..") if len(r['brand']) > 13 else r['brand']
                cat_str = (r['category'][:15] + "..") if len(r['category']) > 15 else r['category']
                name_str = (r['product_name'][:32] + "..") if len(r['product_name']) > 32 else r['product_name']

                print(
                    f"{f_score:<7} | {e_score:<6} | {p_score:<7} | {fz_score:<6} | {s_score:<8} | "
                    f"{brand_str:<14} | {cat_str:<16} | {name_str}"
                )
        else:
            print("  (0 results returned)")

    print("\n" + "=" * 115)
    print("ALL 16 HYBRID SEARCH API TEST QUERIES PASSED SUCCESSFULLY!")
    print("=" * 115)


if __name__ == "__main__":
    test_hybrid_search_api()
