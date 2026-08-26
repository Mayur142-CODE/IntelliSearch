"""Test script for autocomplete HTTP endpoints"""
import requests
import json
import urllib.parse

tests = [
    'phone',
    'phone under',
    'lapt',
    'nyk',
    'spik',
    'wireless head',
    'laptop above',
    'phone below',
    'hedphon',
    'sam',
    'nik',
    'phone under 2',
]

all_passed = True
for q in tests:
    url = f"http://localhost:8000/autocomplete?q={urllib.parse.quote(q)}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"FAIL: {q!r} -> status {r.status_code}: {r.text}")
        all_passed = False
        continue
    data = r.json()
    suggestions = [f"{s['text']} ({s['type']})" for s in data['suggestions']]
    print(f"=== {q!r} === [Status 200]")
    for s in suggestions[:6]:
        print(f"  {s}")
    print()

print(f"OVERALL STATUS: {'ALL PASSED (100%)' if all_passed else 'SOME FAILED'}")
