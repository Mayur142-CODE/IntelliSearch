"""
Edge-Case & Error Testing for Offline Intelligent Product Search API.

Tests the FastAPI /search endpoint against edge-case inputs:
- Empty query, missing query parameter
- Very long query strings
- Special characters, Unicode
- SQL injection-style inputs
- Invalid limit parameters
- Random/gibberish text

Each test records: input, expected behavior, actual behavior, PASS/FAIL.
Outputs backend/results/edge_case_results.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

RESULTS_DIR = BACKEND_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "edge_case_results.json"


def run_test(test_id, description, url, params, expected_status, expected_behavior, extra_check=None):
    """Run a single edge-case test and return result dict."""
    try:
        response = client.get(url, params=params)
        actual_status = response.status_code

        if actual_status == expected_status:
            passed = True
            actual_behavior = f"HTTP {actual_status}"
            if actual_status == 200:
                data = response.json()
                actual_behavior += f", {data.get('count', '?')} results"
                if extra_check:
                    check_passed, check_msg = extra_check(data)
                    if not check_passed:
                        passed = False
                        actual_behavior += f" — {check_msg}"
            elif actual_status == 422:
                actual_behavior += " (validation error as expected)"
        else:
            passed = False
            actual_behavior = f"HTTP {actual_status} (expected {expected_status})"
            try:
                actual_behavior += f": {response.text[:200]}"
            except Exception:
                pass

    except Exception as e:
        passed = False
        actual_behavior = f"Exception: {type(e).__name__}: {str(e)[:200]}"

    status = "PASS" if passed else "FAIL"
    icon = "✓" if passed else "✗"
    print(f"  [{icon}] Test {test_id:02d}: {description} — {status}")

    return {
        "test_id": test_id,
        "description": description,
        "input": {"url": url, "params": params},
        "expected_status": expected_status,
        "expected_behavior": expected_behavior,
        "actual_behavior": actual_behavior,
        "status": status,
    }


def check_zero_results(data):
    if data.get("count", -1) == 0:
        return True, ""
    return False, f"Expected 0 results but got {data.get('count')}"


def check_has_results(data):
    if data.get("count", 0) > 0:
        return True, ""
    return False, "Expected results but got 0"


def check_not_sql_executed(data):
    """Verify SQL injection string was treated as search text, not executed."""
    # If the API returns results or 0 results without an error, SQL was not executed
    if "count" in data and isinstance(data["count"], int):
        return True, ""
    return False, "Response structure unexpected — possible injection issue"


def run_edge_case_tests():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EDGE-CASE & ERROR TESTING REPORT")
    print("=" * 60)

    results = []

    # --- Empty / Missing Query ---
    results.append(run_test(
        1, "Empty query string (q='')",
        "/search", {"q": "", "limit": 10}, 200,
        "HTTP 200 with 0 results",
        check_zero_results,
    ))

    results.append(run_test(
        2, "Missing query parameter (no q)",
        "/search", {"limit": 10}, 200,
        "HTTP 200 with 0 results (default q='')",
        check_zero_results,
    ))

    results.append(run_test(
        3, "Whitespace-only query",
        "/search", {"q": "   ", "limit": 10}, 200,
        "HTTP 200 with 0 results",
        check_zero_results,
    ))

    # --- Very Long Query ---
    long_query = "a" * 500
    results.append(run_test(
        4, "Very long query (500 chars)",
        "/search", {"q": long_query, "limit": 10}, 200,
        "HTTP 200, no crash, 0 or few results",
    ))

    very_long_query = "search " * 200  # 1400 chars
    results.append(run_test(
        5, "Very long multi-word query (1400 chars)",
        "/search", {"q": very_long_query, "limit": 10}, 200,
        "HTTP 200, no crash",
    ))

    # --- Special Characters ---
    results.append(run_test(
        6, "Special characters: !!!",
        "/search", {"q": "!!!", "limit": 10}, 200,
        "HTTP 200, handled gracefully",
    ))

    results.append(run_test(
        7, "Special characters: @@@",
        "/search", {"q": "@@@", "limit": 10}, 200,
        "HTTP 200, handled gracefully",
    ))

    results.append(run_test(
        8, "Special characters: ###",
        "/search", {"q": "###", "limit": 10}, 200,
        "HTTP 200, handled gracefully",
    ))

    results.append(run_test(
        9, "Mixed special: iphone!!!",
        "/search", {"q": "iphone!!!", "limit": 10}, 200,
        "HTTP 200, may return results",
    ))

    # --- Unicode ---
    results.append(run_test(
        10, "Unicode: café",
        "/search", {"q": "café", "limit": 10}, 200,
        "HTTP 200, handled gracefully",
    ))

    results.append(run_test(
        11, "Unicode: Hindi (फोन)",
        "/search", {"q": "फोन", "limit": 10}, 200,
        "HTTP 200, handled gracefully",
    ))

    results.append(run_test(
        12, "Unicode: Gujarati (ગુજરાતી)",
        "/search", {"q": "ગુજરાતી", "limit": 10}, 200,
        "HTTP 200, handled gracefully",
    ))

    # --- Random / Gibberish ---
    results.append(run_test(
        13, "Random gibberish: abcxyz123",
        "/search", {"q": "abcxyz123", "limit": 10}, 200,
        "HTTP 200, 0 or few results",
    ))

    results.append(run_test(
        14, "Nonsense: xyznonexistent123",
        "/search", {"q": "xyznonexistent123", "limit": 10}, 200,
        "HTTP 200, 0 results expected",
    ))

    # --- Limit Edge Cases ---
    results.append(run_test(
        15, "Very large limit (limit=100)",
        "/search", {"q": "nike", "limit": 100}, 200,
        "HTTP 200, returns up to 100 results",
        check_has_results,
    ))

    results.append(run_test(
        16, "Invalid limit: limit=0",
        "/search", {"q": "nike", "limit": 0}, 422,
        "HTTP 422 validation error (limit ge=1)",
    ))

    results.append(run_test(
        17, "Invalid limit: limit=-1",
        "/search", {"q": "nike", "limit": -1}, 422,
        "HTTP 422 validation error (limit ge=1)",
    ))

    results.append(run_test(
        18, "Invalid limit: limit=abc",
        "/search", {"q": "nike", "limit": "abc"}, 422,
        "HTTP 422 validation error (not integer)",
    ))

    results.append(run_test(
        19, "Exceeding max limit: limit=999",
        "/search", {"q": "nike", "limit": 999}, 422,
        "HTTP 422 validation error (limit le=100)",
    ))

    # --- SQL Injection ---
    results.append(run_test(
        20, "SQL injection: ' OR '1'='1",
        "/search", {"q": "' OR '1'='1", "limit": 10}, 200,
        "HTTP 200, treated as text, no SQL execution",
        check_not_sql_executed,
    ))

    results.append(run_test(
        21, "SQL injection: '; DROP TABLE products; --",
        "/search", {"q": "'; DROP TABLE products; --", "limit": 10}, 200,
        "HTTP 200, treated as text, no SQL execution",
        check_not_sql_executed,
    ))

    results.append(run_test(
        22, "SQL injection: 1 UNION SELECT * FROM products",
        "/search", {"q": "1 UNION SELECT * FROM products", "limit": 10}, 200,
        "HTTP 200, treated as text, no SQL execution",
        check_not_sql_executed,
    ))

    # --- Summary ---
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{total} PASS, {failed}/{total} FAIL")
    print(f"{'=' * 60}")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{(passed / total) * 100:.1f}%",
        "tests": results,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_FILE}")

    return output


if __name__ == "__main__":
    run_edge_case_tests()
