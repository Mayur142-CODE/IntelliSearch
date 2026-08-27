"""
Comprehensive Autocomplete & Search Integration Test Suite
Verifies:
1. Multi-token typo correction (nykks shoss -> Nike shoes as #1 Did you mean)
2. Negative test: rejection of unrelated brands (e.g. Uniqlo on nykks shoss)
3. Diverse test inputs: single-word, multi-word, prefixes, prices, natural language
4. Zero regressions in full hybrid search on Enter
"""

import sys
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE AUTOCOMPLETE TEST SUITE (35+ QUERIES)")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    def assert_test(cond: bool, desc: str, detail: str = ""):
        nonlocal tests_passed, tests_failed
        if cond:
            print(f"  [PASS] {desc}")
            if detail:
                print(f"         -> {detail}")
            tests_passed += 1
        else:
            print(f"  [FAIL] {desc}")
            if detail:
                print(f"         -> {detail}")
            tests_failed += 1

    # -------------------------------------------------------------
    # 1. CORE BUG TEST: 'nykks shoss'
    # -------------------------------------------------------------
    print("\n>>> 1. CORE USER BUG TEST: 'nykks shoss'")
    res = requests.get(f"{BASE_URL}/autocomplete?q=nykks+shoss&limit=8")
    assert_test(res.status_code == 200, "HTTP 200 for 'nykks shoss'")
    data = res.json()
    sugs = [s["text"] for s in data.get("suggestions", [])]
    corr_items = [s for s in data.get("suggestions", []) if s.get("is_correction")]

    print(f"       Suggestions: {sugs}")
    assert_test(len(sugs) > 0 and sugs[0].lower() == "nike shoes", "'nykks shoss' top suggestion is 'Nike shoes'", f"Got: {sugs[0] if sugs else None}")
    assert_test(len(corr_items) > 0 and corr_items[0]["text"].lower() == "nike shoes", "'Nike shoes' is flagged with is_correction=True")
    
    # Negative test: Uniqlo or unrelated brand must NOT be in top 3
    top3_lower = [s.lower() for s in sugs[:3]]
    has_uniqlo = any("uniqlo" in s for s in top3_lower)
    assert_test(not has_uniqlo, "No 'Uniqlo' in top suggestions for 'nykks shoss'", f"Top 3: {sugs[:3]}")

    # -------------------------------------------------------------
    # 2. MULTI-WORD TYPOS
    # -------------------------------------------------------------
    print("\n>>> 2. MULTI-WORD TYPOS")
    multi_typos = [
        ("nyke shose", "nike shoes"),
        ("samsng phne", "samsung phone"),
        ("wirless hedphnes", "wireless headphones"),
        ("blutooth hedphones", "bluetooth headphones"),
        ("premum head", "premium headphones"),
    ]
    for raw, expected_target in multi_typos:
        r = requests.get(f"{BASE_URL}/autocomplete", params={"q": raw, "limit": 6})
        items = [s["text"].lower() for s in r.json().get("suggestions", [])]
        matched = any(expected_target in s or s in expected_target for s in items)
        assert_test(matched, f"'{raw}' suggests '{expected_target}'", f"Top: {items[:3]}")

    # -------------------------------------------------------------
    # 3. SINGLE-WORD TYPOS
    # -------------------------------------------------------------
    print("\n>>> 3. SINGLE-WORD TYPOS")
    single_typos = [
        ("lptap", "laptop"),
        ("backpac", "backpack"),
        ("headphnes", "headphones"),
        ("samsng", "samsung"),
        ("nyke", "nike"),
        ("addidas", "adidas"),
        ("pumma", "puma"),
    ]
    for raw, expected_target in single_typos:
        r = requests.get(f"{BASE_URL}/autocomplete", params={"q": raw, "limit": 6})
        items = [s["text"].lower() for s in r.json().get("suggestions", [])]
        matched = any(expected_target in s for s in items)
        assert_test(matched, f"'{raw}' suggests '{expected_target}'", f"Top: {items[:3]}")

    # -------------------------------------------------------------
    # 4. PREFIX / INCOMPLETE QUERIES
    # -------------------------------------------------------------
    print("\n>>> 4. PREFIX & INCOMPLETE TYPING")
    prefixes = [
        ("lap", "laptop"),
        ("back", "backpack"),
        ("wireles hea", "wireless headphones"),
        ("sams pho", "samsung"),
    ]
    for raw, expected_part in prefixes:
        r = requests.get(f"{BASE_URL}/autocomplete", params={"q": raw, "limit": 6})
        items = [s["text"].lower() for s in r.json().get("suggestions", [])]
        matched = any(expected_part in s for s in items)
        assert_test(matched, f"Prefix '{raw}' produces completions with '{expected_part}'", f"Top: {items[:3]}")

    # -------------------------------------------------------------
    # 5. DYNAMIC PRICE AUTOCOMPLETE
    # -------------------------------------------------------------
    print("\n>>> 5. DYNAMIC PRICE AUTOCOMPLETE")
    price_queries = [
        "phone unde",
        "phone below",
        "phone above",
        "laptop under 2",
    ]
    for pq in price_queries:
        r = requests.get(f"{BASE_URL}/autocomplete", params={"q": pq, "limit": 6})
        items = [s["text"] for s in r.json().get("suggestions", [])]
        assert_test(len(items) > 0 and any(any(c.isdigit() for c in s) for s in items), f"'{pq}' returns dynamic price completions", f"Results: {items[:3]}")

    # -------------------------------------------------------------
    # 6. NATURAL LANGUAGE CONCEPTUAL QUERIES
    # -------------------------------------------------------------
    print("\n>>> 6. NATURAL LANGUAGE QUERIES")
    nl_queries = [
        ("something to carry my laptop", ["laptop", "bag", "backpack", "travel"]),
        ("device for listening to music", ["sound", "audio", "headphone", "speaker", "anker"]),
        ("shoes for morning walk", ["shoes", "running", "sneaker"]),
        ("something to charge my phone", ["charge", "charger", "cable", "otterbox"]),
        ("bag for traveling", ["travel", "bag", "duffel", "backpack"]),
    ]
    for q_nl, expected_keywords in nl_queries:
        r = requests.get(f"{BASE_URL}/autocomplete", params={"q": q_nl, "limit": 6})
        items = [s["text"].lower() for s in r.json().get("suggestions", [])]
        matched = any(any(k in s for k in expected_keywords) for s in items)
        assert_test(matched, f"'{q_nl}' returns relevant semantic suggestions", f"Top: {items[:3]}")

    # -------------------------------------------------------------
    # 7. UNSEEN / DIVERSE QUERIES (GENERALIZATION)
    # -------------------------------------------------------------
    print("\n>>> 7. UNSEEN / DIVERSE QUERIES")
    unseen_queries = [
        "adidass runing shoos",
        "gming keybord",
        "wireles speker",
        "sam sung ultra",
        "portable pwer bnk",
        "camera tripod stnd",
        "winter cot",
        "denm jens",
    ]
    for uq in unseen_queries:
        r = requests.get(f"{BASE_URL}/autocomplete", params={"q": uq, "limit": 5})
        items = [s["text"] for s in r.json().get("suggestions", [])]
        assert_test(r.status_code == 200 and len(items) > 0, f"Unseen query '{uq}' returns suggestions without error", f"Top: {items[:2]}")

    # -------------------------------------------------------------
    # 8. FULL SEARCH VERIFICATION ON ENTER (PRESERVATION OF SEARCH)
    # -------------------------------------------------------------
    print("\n>>> 8. FULL SEARCH ON ENTER")
    s_res = requests.get(f"{BASE_URL}/search", params={"q": "nykks shoss", "limit": 5})
    assert_test(s_res.status_code == 200, "Full search /search?q=nykks shoss returns 200 OK")
    s_data = s_res.json()
    results = s_data.get("results", [])
    assert_test(len(results) > 0, f"Full search returned {len(results)} results")
    brands = [p.get("brand") for p in results]
    assert_test("Nike" in brands, f"Nike is among top search results for 'nykks shoss': {brands}")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"TOTAL: {tests_passed + tests_failed} | PASSED: {tests_passed} | FAILED: {tests_failed}")
    print("=" * 70)
    if tests_failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
