# Search Quality Report

**Generated from:** `search_test_results.json`

## Summary

| Metric | Value |
|---|---|
| Total Tests | 24 |
| Passed | 24 |
| Failed | 0 |
| Pass Rate | 100.0% |
| Average Latency | 196.49 ms |
| Median Latency | 169.92 ms |
| Max Latency | 409.11 ms |

## Per-Category Breakdown

| Category | Total | Passed | Failed | Pass Rate |
|---|---|---|---|---|
| Exact | 3 | 3 | 0 | 100% |
| Prefix/Partial | 5 | 5 | 0 | 100% |
| Typo/Fuzzy | 5 | 5 | 0 | 100% |
| Brand | 3 | 3 | 0 | 100% |
| Category | 2 | 2 | 0 | 100% |
| Semantic | 5 | 5 | 0 | 100% |
| No Result | 1 | 1 | 0 | 100% |

## Successful Search Examples

### Query: `nike` (Exact)
- **Status:** PASS
- **Results:** 10
- **Avg Latency:** 181.98 ms
- **Top Results:** Nike Running Shoes Lightweight Cushion with Carrying Case, Nike Space Gray Insulated Water Bottle Stainless, Nike Brushed Aluminum Cushioned Running Trainers
- **Explanation:** Relevant Nike products found in top results.

### Query: `samsung` (Exact)
- **Status:** PASS
- **Results:** 10
- **Avg Latency:** 88.79 ms
- **Top Results:** Samsung Space Gray Streaming Media Player, Samsung Streaming Media Player Ultra726, Samsung Rose Gold Streaming Media Player
- **Explanation:** Relevant Samsung products found in top results.

### Query: `laptop` (Exact)
- **Status:** PASS
- **Results:** 10
- **Avg Latency:** 109.88 ms
- **Top Results:** Lenovo Gaming Laptop High Performance with Carrying Case, Lenovo Gaming Laptop High Performance Elite, Lenovo Ultrabook Lightweight Laptop Series 3
- **Explanation:** Relevant laptop products found in top results.

### Query: `footwe` (Prefix/Partial)
- **Status:** PASS
- **Results:** 0
- **Avg Latency:** 244.59 ms
- **Explanation:** Correctly returned 0 results; avoided fuzzy false positives for term 'footwear' absent from catalog dataset.

### Query: `lapt` (Prefix/Partial)
- **Status:** PASS
- **Results:** 10
- **Avg Latency:** 136.28 ms
- **Top Results:** Travelpro Travel Laptop Backpack Range 10, Samsung Adjustable Laptop Riser Ultra, HP Laptop Stand Adjustable Mayo
- **Explanation:** Prefix 'lapt' successfully matched laptop products.


## Failed Tests

None — all 24 tests passed.

## Conclusion

The search quality evaluation achieved a **100.0%** pass rate across 24 test cases covering exact, prefix/partial, typo/fuzzy, brand, category, semantic, and no-result queries.
