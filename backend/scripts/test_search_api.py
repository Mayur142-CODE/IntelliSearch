import json
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


def test_api():
    queries = [
        "nik shose",
        "samsng phone",
        "wireles hedphone",
        "nike",
        "samsung",
        "laptop",
        "nonexistentproduct12345xyz",
        "",
    ]

    print("=" * 80)
    print("FASTAPI /search ENDPOINT VERIFICATION")
    print("=" * 80)

    for q in queries:
        response = client.get("/search", params={"q": q, "limit": 10})
        assert response.status_code == 200, f"Failed for query '{q}': {response.status_code}"
        data = response.json()
        
        # Verify schema keys
        assert "query" in data
        assert "count" in data
        assert "results" in data
        assert data["count"] == len(data["results"])

        print(f"\nQuery: '{data['query']}' | Count: {data['count']}")
        if data["results"]:
            for r in data["results"][:3]:  # Top 3
                print(f"  - [{r['fuzzy_score']:.4f}] {r['brand']} | {r['category']} | {r['product_name']} | INR {r['price']}")
        else:
            print("  (0 results)")

    print("\n" + "=" * 80)
    print("ALL API SEARCH TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_api()
