"""
Comprehensive HTTP test suite for autocomplete & search
"""
import requests
import json
import urllib.parse
import time

test_queries = [
    "nykes shoos",
    "nyke shoes",
    "nkie shoes",
    "nik shoes",
    "lptap",
    "laptop unde",
    "wirless hedphones",
    "headphnes",
    "backpac",
    "samsng phone",
    "phone unde",
    "phone below",
    "phone above",
    "phone between",
    "wireless head",
    "budget lap",
    "premium head",
    "something to carry my laptop",
    "device for listening to music",
    "shoes for morning walk",
    "something to charge my phone",
    "bag for traveling",
]

def main():
    print("=" * 60)
    print("AUTOCOMPLETE HTTP ENDPOINT TESTS")
    print("=" * 60)

    for q in test_queries:
        url = f"http://127.0.0.1:8000/autocomplete?q={urllib.parse.quote(q)}"
        t0 = time.perf_counter()
        res = requests.get(url)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if res.status_code != 200:
            print(f"FAILED: {q!r} -> status {res.status_code}")
            continue

        data = res.json()
        suggestions = [s['text'] for s in data.get('suggestions', [])]
        print(f"Query: {q!r} ({elapsed_ms:.1f} ms)")
        for i, s in enumerate(suggestions[:6], 1):
            print(f"  {i}. {s}")
        print()

    print("=" * 60)
    print("VERIFYING FULL SEARCH FOR 'nykes shoos'")
    print("=" * 60)
    search_url = f"http://127.0.0.1:8000/search?q={urllib.parse.quote('nykes shoos')}&limit=5"
    t0 = time.perf_counter()
    s_res = requests.get(search_url)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    s_data = s_res.json()
    print(f"Search status: {s_res.status_code} ({elapsed_ms:.1f} ms)")
    print(f"Count: {s_data.get('count', 0)}")
    for p in s_data.get('results', [])[:4]:
        print(f"  - [{p.get('brand')}] {p.get('product_name')} (Rs. {p.get('price')})")

if __name__ == "__main__":
    main()
