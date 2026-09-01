"""
Score Distribution Diagnostic — Collects raw signal scores across relevant,
irrelevant, and edge-case queries to empirically determine threshold values.

Run inside container:  python scripts/score_diagnostics.py
"""

import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.services.query_parser import parse_query
from app.services.search_ranking import search_products
from app.services.semantic_search import semantic_search_products
from app.services.fuzzy_search import fuzzy_search_products

# =========================================================================
# Query sets — each labeled with expected relevance
# =========================================================================

RELEVANT_QUERIES = [
    # Queries that SHOULD return meaningful results from the catalog
    "nike shoes",
    "gaming laptop",
    "wireless headphones",
    "sony headphones",
    "logitech mouse",
    "samsung phone case",
    "running shoes",
    "bluetooth speaker",
    "laptop bag",
    "mechanical keyboard",
    "yoga mat",
    "water bottle",
    "desk lamp",
    "office chair",
    "backpack",
]

IRRELEVANT_QUERIES = [
    # Queries that have NO matching products — should return empty or very weak
    "something to eat or drink",
    "recipe for chocolate cake",
    "weather forecast tomorrow",
    "how to learn python programming",
    "best restaurants near me",
    "flight tickets to paris",
    "asdkjhaskjdh",
    "zzxqwvlkmnb",
    "cat food organic grain free",
    "mortgage calculator",
]

TYPO_QUERIES = [
    # Queries with typos that SHOULD still match (via correction pipeline)
    ("nykyeyy shoes", "Nike shoes"),
    ("nyky shoes", "Nike shoes"),
    ("nyyskee", "Nike"),
    ("smsng phone", "Samsung phone"),
    ("logitec mouse", "Logitech mouse"),
]

# =========================================================================
# Data collection
# =========================================================================

def collect_full_diagnostics(db):
    """Collect per-candidate score breakdowns for all query sets."""
    
    all_data = {}

    # --- Relevant queries ---
    print("=" * 80)
    print("RELEVANT QUERIES — Score Distributions")
    print("=" * 80)
    
    relevant_scores = defaultdict(list)  # signal_name -> [score, ...]
    
    for q in RELEVANT_QUERIES:
        results, parsed = search_products(db, q, limit=10, min_final_score=0.0)
        print(f"\n  Query: {q!r} -> {len(results)} results (semantic_query={parsed.semantic_query!r})")
        
        for r in results[:5]:
            print(f"    [{r.final_score:.4f}] {r.product.product_name[:50]:50s} "
                  f"e={r.exact_score:.3f} p={r.partial_score:.3f} "
                  f"f={r.fuzzy_score:.3f} s={r.semantic_score:.3f} "
                  f"src={','.join(r.candidate_sources)}")
            relevant_scores["final"].append(r.final_score)
            relevant_scores["semantic"].append(r.semantic_score)
            relevant_scores["fuzzy"].append(r.fuzzy_score)
            relevant_scores["exact"].append(r.exact_score)
            relevant_scores["partial"].append(r.partial_score)
    
    # --- Irrelevant queries ---
    print("\n" + "=" * 80)
    print("IRRELEVANT QUERIES — Score Distributions (should be empty or very weak)")
    print("=" * 80)
    
    irrelevant_scores = defaultdict(list)
    
    for q in IRRELEVANT_QUERIES:
        # Use min_final_score=0.0 to see everything, even junk
        results, parsed = search_products(db, q, limit=10, min_final_score=0.0)
        print(f"\n  Query: {q!r} -> {len(results)} candidates at score>=0.0")
        
        for r in results[:5]:
            print(f"    [{r.final_score:.4f}] {r.product.product_name[:50]:50s} "
                  f"e={r.exact_score:.3f} p={r.partial_score:.3f} "
                  f"f={r.fuzzy_score:.3f} s={r.semantic_score:.3f} "
                  f"src={','.join(r.candidate_sources)} "
                  f"brand_match={r.brand_match} cat_match={r.category_match}")
            irrelevant_scores["final"].append(r.final_score)
            irrelevant_scores["semantic"].append(r.semantic_score)
            irrelevant_scores["fuzzy"].append(r.fuzzy_score)
            irrelevant_scores["exact"].append(r.exact_score)
            irrelevant_scores["partial"].append(r.partial_score)

    # --- Raw semantic scores for irrelevant queries (no ranking filter) ---
    print("\n" + "=" * 80)
    print("RAW SEMANTIC SCORES — Irrelevant queries (ChromaDB nearest neighbors)")
    print("=" * 80)
    
    raw_semantic_irrelevant = []
    for q in IRRELEVANT_QUERIES:
        sem_results = semantic_search_products(db=db, query=q, limit=10, min_similarity=0.0)
        scores = [float(r.semantic_score) for r in sem_results]
        raw_semantic_irrelevant.extend(scores)
        if scores:
            print(f"  {q!r}: top={max(scores):.4f}, min={min(scores):.4f}, "
                  f"mean={statistics.mean(scores):.4f}, n={len(scores)}")
        else:
            print(f"  {q!r}: 0 semantic candidates")
    
    raw_semantic_relevant = []
    for q in RELEVANT_QUERIES:
        sem_results = semantic_search_products(db=db, query=q, limit=10, min_similarity=0.0)
        scores = [float(r.semantic_score) for r in sem_results]
        raw_semantic_relevant.extend(scores)
        if scores:
            print(f"  {q!r}: top={max(scores):.4f}, min={min(scores):.4f}, "
                  f"mean={statistics.mean(scores):.4f}, n={len(scores)}")

    # --- Typo queries ---
    print("\n" + "=" * 80)
    print("TYPO QUERIES — Correction verification + scores")
    print("=" * 80)
    
    for q, expected in TYPO_QUERIES:
        parsed = parse_query(db, q)
        results, _ = search_products(db, q, limit=5, min_final_score=0.0)
        corrected_ok = expected.lower() in parsed.semantic_query.lower()
        print(f"\n  Query: {q!r}")
        print(f"    semantic_query: {parsed.semantic_query!r} (expected ~{expected!r}) {'✓' if corrected_ok else '✗'}")
        print(f"    did_you_mean: {parsed.did_you_mean!r}")
        print(f"    brands: {parsed.detected_brands}")
        for r in results[:3]:
            print(f"    [{r.final_score:.4f}] {r.product.product_name[:50]:50s} "
                  f"f={r.fuzzy_score:.3f} s={r.semantic_score:.3f}")

    # --- Summary statistics ---
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for label, scores in [("RELEVANT", relevant_scores), ("IRRELEVANT", irrelevant_scores)]:
        print(f"\n  {label}:")
        for signal in ["final", "semantic", "fuzzy", "exact", "partial"]:
            vals = scores.get(signal, [])
            if vals:
                print(f"    {signal:10s}: n={len(vals):3d}  "
                      f"min={min(vals):.4f}  p10={sorted(vals)[len(vals)//10]:.4f}  "
                      f"median={statistics.median(vals):.4f}  "
                      f"mean={statistics.mean(vals):.4f}  "
                      f"p90={sorted(vals)[9*len(vals)//10]:.4f}  "
                      f"max={max(vals):.4f}")
            else:
                print(f"    {signal:10s}: no data")

    print(f"\n  RAW SEMANTIC (ChromaDB) — RELEVANT vs IRRELEVANT:")
    if raw_semantic_relevant:
        print(f"    Relevant  : n={len(raw_semantic_relevant):3d}  "
              f"min={min(raw_semantic_relevant):.4f}  "
              f"median={statistics.median(raw_semantic_relevant):.4f}  "
              f"mean={statistics.mean(raw_semantic_relevant):.4f}  "
              f"max={max(raw_semantic_relevant):.4f}")
    if raw_semantic_irrelevant:
        print(f"    Irrelevant: n={len(raw_semantic_irrelevant):3d}  "
              f"min={min(raw_semantic_irrelevant):.4f}  "
              f"median={statistics.median(raw_semantic_irrelevant):.4f}  "
              f"mean={statistics.mean(raw_semantic_irrelevant):.4f}  "
              f"max={max(raw_semantic_irrelevant):.4f}")

    # --- Gap analysis ---
    print("\n  THRESHOLD GAP ANALYSIS:")
    if relevant_scores["final"] and irrelevant_scores["final"]:
        rel_min = min(relevant_scores["final"])
        irr_max = max(irrelevant_scores["final"])
        print(f"    Final score: relevant_min={rel_min:.4f}, irrelevant_max={irr_max:.4f}, "
              f"gap={rel_min - irr_max:.4f}")
    if raw_semantic_relevant and raw_semantic_irrelevant:
        rel_p10 = sorted(raw_semantic_relevant)[len(raw_semantic_relevant)//10]
        irr_p90 = sorted(raw_semantic_irrelevant)[9*len(raw_semantic_irrelevant)//10]
        irr_max_s = max(raw_semantic_irrelevant)
        rel_min_s = min(raw_semantic_relevant)
        print(f"    Semantic raw: relevant_min={rel_min_s:.4f}, relevant_p10={rel_p10:.4f}, "
              f"irrelevant_p90={irr_p90:.4f}, irrelevant_max={irr_max_s:.4f}")
        print(f"    Semantic gap (rel_p10 - irr_p90): {rel_p10 - irr_p90:.4f}")

    # --- Strong signal distribution for irrelevant ---
    print("\n  STRONG SIGNAL ANALYSIS (irrelevant queries):")
    print(f"    Current STRONG_SIGNAL_SEMANTIC_MIN = 0.40")
    sem_above_40 = [s for s in irrelevant_scores["semantic"] if s >= 0.40]
    sem_above_45 = [s for s in irrelevant_scores["semantic"] if s >= 0.45]
    sem_above_50 = [s for s in irrelevant_scores["semantic"] if s >= 0.50]
    sem_above_55 = [s for s in irrelevant_scores["semantic"] if s >= 0.55]
    print(f"    Irrelevant candidates with semantic >= 0.40: {len(sem_above_40)}")
    print(f"    Irrelevant candidates with semantic >= 0.45: {len(sem_above_45)}")
    print(f"    Irrelevant candidates with semantic >= 0.50: {len(sem_above_50)}")
    print(f"    Irrelevant candidates with semantic >= 0.55: {len(sem_above_55)}")
    
    rel_sem_above_40 = [s for s in relevant_scores["semantic"] if s >= 0.40]
    rel_sem_above_45 = [s for s in relevant_scores["semantic"] if s >= 0.45]
    rel_sem_above_50 = [s for s in relevant_scores["semantic"] if s >= 0.50]
    rel_sem_above_55 = [s for s in relevant_scores["semantic"] if s >= 0.55]
    print(f"    Relevant candidates with semantic >= 0.40: {len(rel_sem_above_40)}")
    print(f"    Relevant candidates with semantic >= 0.45: {len(rel_sem_above_45)}")
    print(f"    Relevant candidates with semantic >= 0.50: {len(rel_sem_above_50)}")
    print(f"    Relevant candidates with semantic >= 0.55: {len(rel_sem_above_55)}")

    # --- What happens at various final score thresholds ---
    print("\n  FINAL SCORE THRESHOLD SWEEP:")
    for t in [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25]:
        rel_pass = len([s for s in relevant_scores["final"] if s >= t])
        irr_pass = len([s for s in irrelevant_scores["final"] if s >= t])
        rel_total = len(relevant_scores["final"])
        irr_total = len(irrelevant_scores["final"])
        print(f"    threshold={t:.2f}: relevant_pass={rel_pass}/{rel_total}  "
              f"irrelevant_pass={irr_pass}/{irr_total}")

    # --- MIN_FINAL_SCORE_WEAK sweep (only applies when no strong signal) ---
    print("\n  WEAK-SIGNAL THRESHOLD SWEEP (candidates with NO strong signal only):")
    # Re-collect with strong signal flags
    weak_rel = []
    weak_irr = []
    for q in RELEVANT_QUERIES:
        results, _ = search_products(db, q, limit=10, min_final_score=0.0)
        for r in results:
            has_strong = (r.exact_score > 0 or r.partial_score > 0 or 
                         r.fuzzy_score >= 0.30 or r.semantic_score >= 0.40 or
                         r.brand_match)
            if not has_strong:
                weak_rel.append(r.final_score)
    for q in IRRELEVANT_QUERIES:
        results, _ = search_products(db, q, limit=10, min_final_score=0.0)
        for r in results:
            has_strong = (r.exact_score > 0 or r.partial_score > 0 or 
                         r.fuzzy_score >= 0.30 or r.semantic_score >= 0.40 or
                         r.brand_match)
            if not has_strong:
                weak_irr.append(r.final_score)
    
    print(f"    Weak-signal relevant candidates: {len(weak_rel)}")
    print(f"    Weak-signal irrelevant candidates: {len(weak_irr)}")
    if weak_rel:
        print(f"    Weak relevant: min={min(weak_rel):.4f} max={max(weak_rel):.4f} "
              f"mean={statistics.mean(weak_rel):.4f}")
    if weak_irr:
        print(f"    Weak irrelevant: min={min(weak_irr):.4f} max={max(weak_irr):.4f} "
              f"mean={statistics.mean(weak_irr):.4f}")
    for t in [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20]:
        wr_pass = len([s for s in weak_rel if s >= t])
        wi_pass = len([s for s in weak_irr if s >= t])
        print(f"    threshold={t:.2f}: weak_relevant_pass={wr_pass}/{len(weak_rel)}  "
              f"weak_irrelevant_pass={wi_pass}/{len(weak_irr)}")


if __name__ == "__main__":
    db = SessionLocal()
    try:
        collect_full_diagnostics(db)
    finally:
        db.close()
