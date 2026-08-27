"""
Test script for the new complete-query autocomplete engine
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import SessionLocal
from app.services.autocomplete import generate_suggestions

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
    db = SessionLocal()
    print("Running autocomplete tests...\n")
    for q in test_queries:
        suggestions = generate_suggestions(db, q, max_results=6)
        texts = [s.text for s in suggestions]
        print(f"Query: {q!r}")
        for i, t in enumerate(texts, 1):
            print(f"  {i}. {t}")
        print()

if __name__ == "__main__":
    main()
