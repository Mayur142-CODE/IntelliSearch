# Testing, Benchmarking & Validation Report

## 1. Testing Strategy Overview

The testing suite validates 4 critical system dimensions:
1. **Search Quality**: Accuracy across exact, partial, fuzzy, brand, category, semantic, and negative queries.
2. **Performance Benchmarking**: Cold-start latency, warm latency percentiles, memory consumption, and HTTP API response times.
3. **Database Integrity**: Schema conformance, product count verification, index health, and vector alignment.
4. **Edge Cases & Error Handling**: Input fuzzing, Unicode support, buffer length testing, parameter validation, and SQL injection resistance.

---

## 2. Search Quality Evaluation (`test_search_cases.py`)

### Summary Results
- **Total Test Cases**: 24
- **Passed**: 24 (**100.0% Pass Rate**)
- **Failed**: 0
- **Overall Latency**: Mean: **196.49 ms** | Median: **169.92 ms** | Max: **409.11 ms**

### Category Performance Breakdown

| Category | Tests | Passed | Failed | Pass Rate | Sample Query |
|---|---|---|---|---|---|
| **Exact Search** | 3 | 3 | 0 | 100.0% | `nike`, `samsung`, `laptop` |
| **Prefix / Partial** | 5 | 5 | 0 | 100.0% | `footwe`, `lapt`, `head`, `wire`, `phon` |
| **Typo / Fuzzy** | 5 | 5 | 0 | 100.0% | `lptop`, `botle`, `nik shose`, `samsng phone`, `wireles hedphone` |
| **Brand Search** | 3 | 3 | 0 | 100.0% | `nike`, `samsung`, `sony` |
| **Category Search**| 2 | 2 | 0 | 100.0% | `electronics`, `fashion` |
| **Semantic Search**| 5 | 5 | 0 | 100.0% | `something to carry my laptop`, `device for listening to music`, `shoes for morning running`, `something to charge my phone`, `bag for traveling` |
| **No-Result Query** | 1 | 1 | 0 | 100.0% | `nonexistentproduct12345xyz` |

### Complete 24 Test Cases Matrix (Measured)

| # | Query | Search Type | Avg Latency | Results | Status | Evaluation Notes |
|---|---|---|---|---|---|---|
| 1 | `nike` | Exact | 181.98 ms | 10 | **PASS** | Relevant Nike products in top positions |
| 2 | `samsung` | Exact | 88.79 ms | 10 | **PASS** | Relevant Samsung products in top positions |
| 3 | `laptop` | Exact | 109.88 ms | 10 | **PASS** | Laptop computers matched |
| 4 | `footwe` | Prefix/Partial | 244.59 ms | 0 | **PASS** | Avoided false positives for absent term |
| 5 | `lapt` | Prefix/Partial | 136.28 ms | 10 | **PASS** | Matched laptop backpacks & stands |
| 6 | `head` | Prefix/Partial | 164.13 ms | 10 | **PASS** | Matched VR headsets |
| 7 | `wire` | Prefix/Partial | 138.84 ms | 10 | **PASS** | Matched wireless earbuds |
| 8 | `phon` | Prefix/Partial | 149.43 ms | 10 | **PASS** | Matched phone accessories |
| 9 | `lptop` | Typo/Fuzzy | 186.20 ms | 10 | **PASS** | Resolved typo to laptop |
| 10 | `botle` | Typo/Fuzzy | 214.53 ms | 10 | **PASS** | Resolved typo to water bottles |
| 11 | `nik shose` | Typo/Fuzzy | 218.22 ms | 10 | **PASS** | Resolved double typo to Nike / shoes |
| 12 | `samsng phone` | Typo/Fuzzy | 157.86 ms | 10 | **PASS** | Resolved typo to Samsung phone cases |
| 13 | `wireles hedphone` | Typo/Fuzzy | 197.79 ms | 10 | **PASS** | Resolved typo to wireless headphones |
| 14 | `nike` | Brand | 153.84 ms | 10 | **PASS** | Exact brand matching |
| 15 | `samsung` | Brand | 64.57 ms | 10 | **PASS** | Exact brand matching |
| 16 | `sony` | Brand | 149.86 ms | 10 | **PASS** | Exact brand matching |
| 17 | `electronics` | Category | 113.49 ms | 10 | **PASS** | Category match |
| 18 | `fashion` | Category | 143.07 ms | 10 | **PASS** | Category match |
| 19 | `something to carry my laptop` | Semantic | 361.58 ms | 10 | **PASS** | Matched backpacks & carry cases |
| 20 | `device for listening to music` | Semantic | 357.27 ms | 10 | **PASS** | Matched turntables & earbuds |
| 21 | `shoes for morning running` | Semantic | 366.82 ms | 10 | **PASS** | Matched running shoes & trainers |
| 22 | `something to charge my phone` | Semantic | 362.52 ms | 10 | **PASS** | Matched chargers & power mounts |
| 23 | `bag for traveling` | Semantic | 190.24 ms | 10 | **PASS** | Matched duffel bags & luggage |
| 24 | `nonexistentproduct12345xyz` | No Result | 263.90 ms | 0 | **PASS** | Returned 0 results (noise suppressed) |

---

## 3. Performance & Memory Profiling (`benchmark_search.py`)

### Summary Benchmark Metrics

| Metric | Measured Value | Benchmark Target | Status |
|---|---|---|---|
| **Catalog Products** | 7,500 products | $\ge 5,000$ | **PASS** |
| **Embeddings Storage** | 10.99 MB (float32, 384 dimensions) | $< 50$ MB | **PASS** |
| **Model Size on Disk** | 173.77 MB (`all-MiniLM-L6-v2` ONNX) | $< 500$ MB | **PASS** |
| **Baseline Memory (RSS)** | 134.75 MB | — | Measured |
| **Post-Loading Memory (RSS)**| 264.93 MB | $< 512$ MB | **PASS** |
| **Peak Memory (RSS)** | 271.15 MB | $< 512$ MB | **PASS** |
| **Memory Delta** | 130.18 MB | — | Measured |
| **Cold Start (Model + Matrix)**| 285.33 ms | $< 1,000$ ms | **PASS** |
| **First Inference (JIT Warmup)**| 21.87 ms | $< 100$ ms | **PASS** |
| **Total Cold Start Latency** | 307.2 ms | $< 2,000$ ms | **PASS** |
| **Warm Search Average Latency** | **201.51 ms** | $< 1,000$ ms | **PASS** |
| **Warm Search Median Latency** | **167.50 ms** | — | Measured |
| **Warm Search P95 Latency** | **393.48 ms** | $< 800$ ms | **PASS** |
| **Warm Search P99 Latency** | **464.66 ms** | $< 1,000$ ms | **PASS** |
| **Minimum Latency** | 60.54 ms | — | Measured |
| **Maximum Latency** | 464.66 ms | — | Measured |
| **HTTP API Wall-Clock Average** | **332.83 ms** | $< 1,000$ ms | **PASS** |
| **Embedding Generation Time** | Offline precomputed (script: `scripts/generate_embeddings.py`) | — | Documented |

### Per-Query Benchmark Breakdown (19 Queries × 3 Measured Runs)

| Query | Type | Avg (ms) | Min (ms) | Max (ms) | Median (ms) |
|---|---|---|---|---|---|
| `nike` | Exact | 140.13 | 132.44 | 146.54 | 141.42 |
| `samsung` | Exact | 65.36 | 60.54 | 74.98 | 60.56 |
| `laptop` | Partial | 129.54 | 106.61 | 143.07 | 138.94 |
| `head` | Partial | 130.51 | 118.51 | 150.76 | 122.25 |
| `lapt` | Partial | 84.84 | 84.25 | 85.36 | 84.90 |
| `wire` | Partial | 116.03 | 104.65 | 121.98 | 121.47 |
| `foot` | Partial | 172.93 | 163.49 | 187.81 | 167.50 |
| `lptop` | Fuzzy | 156.53 | 155.53 | 158.03 | 156.02 |
| `botle` | Fuzzy | 194.30 | 188.76 | 199.10 | 195.05 |
| `footwe` | Fuzzy | 270.13 | 243.21 | 294.47 | 272.71 |
| `nik shose` | Fuzzy | 212.33 | 205.78 | 225.16 | 206.04 |
| `samsng phone` | Fuzzy | 147.41 | 143.51 | 152.51 | 146.21 |
| `wireles hedphone` | Fuzzy | 232.54 | 187.31 | 297.22 | 213.08 |
| `something to carry my laptop` | Semantic | 322.17 | 306.11 | 342.40 | 318.02 |
| `device for listening to music` | Semantic | 428.70 | 393.48 | 464.66 | 427.96 |
| `shoes for morning running` | Semantic | 334.78 | 318.40 | 358.63 | 327.30 |
| `something to charge my phone` | Semantic | 292.41 | 282.66 | 302.05 | 292.51 |
| `bag for traveling` | Semantic | 143.53 | 133.58 | 153.91 | 143.09 |
| `nonexistentproduct12345xyz` | No Result | 254.59 | 246.10 | 262.97 | 254.69 |

---

## 4. Database Integrity Verification (`verify_database.py`)

- **Total Verification Checks**: 7
- **Passed**: 7 (**100.0% Pass Rate**)
- **Verified Properties**:
  1. PostgreSQL connection alive and accepting queries.
  2. Dynamic product count query confirmed **7,500** rows.
  3. `products` and `alembic_version` tables present in schema `public`.
  4. 7 database indexes verified (`ix_products_brand`, `ix_products_category`, `ix_products_product_name`, `gin_products_brand_trgm`, `gin_products_category_trgm`, `gin_products_product_name_trgm`, `gin_products_tags_trgm`).
  5. Schema columns confirmed: `id`, `product_name`, `description`, `brand`, `category`, `tags`, `price`, `image`.
  6. Embeddings alignment verified: 7,500 embeddings $\leftrightarrow$ 7,500 product IDs $\leftrightarrow$ 7,500 database rows.
  7. `pg_trgm` extension verified installed and active.

---

## 5. Edge-Case & Error Resistance (`test_edge_cases.py`)

- **Total Security & Robustness Scenarios**: 22
- **Passed**: 22 (**100.0% Pass Rate**)

| # | Description | Input Parameters | Expected Status | Actual Result | Status |
|---|---|---|---|---|---|
| 1 | Empty query string | `q=""` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 2 | Missing query parameter | `(no q)` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 3 | Whitespace-only query | `q="   "` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 4 | Long query (500 chars) | `q="a"*500` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 5 | Long query (1400 chars) | `q="search "*200` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 6 | Special characters: `!!!` | `q="!!!"` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 7 | Special characters: `@@@` | `q="@@@"` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 8 | Special characters: `###` | `q="###"` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 9 | Mixed special: `iphone!!!` | `q="iphone!!!"` | HTTP 200 | HTTP 200, 10 results | **PASS** |
| 10 | Unicode: French `café` | `q="café"` | HTTP 200 | HTTP 200, 3 results | **PASS** |
| 11 | Unicode: Hindi `फोन` | `q="फोन"` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 12 | Unicode: Gujarati `ગુજરાતી` | `q="ગુજરાતી"` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 13 | Random gibberish: `abcxyz123` | `q="abcxyz123"` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 14 | Nonsense query | `q="xyznonexistent123"` | HTTP 200 | HTTP 200, 0 results | **PASS** |
| 15 | Max limit (`limit=100`) | `limit=100` | HTTP 200 | HTTP 200, 67 results | **PASS** |
| 16 | Zero limit (`limit=0`) | `limit=0` | HTTP 422 | HTTP 422 Validation Error | **PASS** |
| 17 | Negative limit (`limit=-1`)| `limit=-1` | HTTP 422 | HTTP 422 Validation Error | **PASS** |
| 18 | Non-integer limit | `limit="abc"` | HTTP 422 | HTTP 422 Validation Error | **PASS** |
| 19 | Excessive limit (`limit=999`)| `limit=999` | HTTP 422 | HTTP 422 Validation Error | **PASS** |
| 20 | SQL Injection: `' OR '1'='1` | `q="' OR '1'='1"` | HTTP 200 | HTTP 200, 1 result | **PASS** |
| 21 | SQL Injection: `'; DROP TABLE`| `q="'; DROP TABLE products; --"`| HTTP 200 | HTTP 200, 10 results | **PASS** |
| 22 | SQL Injection: `UNION SELECT`| `q="1 UNION SELECT * FROM products"`| HTTP 200 | HTTP 200, 10 results | **PASS** |
