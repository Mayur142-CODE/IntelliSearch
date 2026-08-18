# Search Engine & Ranking Methodology

## 1. Search Engine Requirements

Product search in e-commerce exhibits diverse user input patterns:
- Exact product names, brands, or SKUs (`nike`, `samsung`, `laptop`)
- Incremental typing / prefixes (`lapt`, `wire`, `phon`)
- Spelling mistakes and typographical errors (`lptop`, `botle`, `nik shose`, `wireles hedphone`)
- Conceptual and intent-based natural language queries (`something to carry my laptop`, `shoes for morning running`)
- Nonsense / non-existent queries (`nonexistentproduct12345xyz`)

A single retrieval mechanism cannot satisfy all patterns:
- Exact search fails completely on typos and descriptive queries.
- Trigram fuzzy search struggles with pure semantic intent (e.g., matching "backpack" when searching "carry my laptop").
- Semantic search can hallucinate false positives on non-matching queries or rank generic items above exact brand matches.

NorthStar resolves this via a **4-Source Hybrid Retrieval & Multi-Signal Scoring Engine**.

---

## 2. The 4-Source Candidate Generation Layer

```
                        Incoming Query String
                                  |
    +-----------------+-----------+-----------+-----------------+
    |                 |                       |                 |
    v                 v                       v                 v
[Source 1]        [Source 2]              [Source 3]        [Source 4]
Exact Match     Prefix / Partial         pg_trgm Fuzzy     FastEmbed Semantic
SQL Equality     SQL ILIKE Patterns     Multi-Path GIN     Vector Dot Product
    |                 |                       |                 |
    +-----------------+-----------+-----------+-----------------+
                                  |
                                  v
                    Candidate Union Map by Product ID
```

### Source 1: Exact Match Candidate Generation
- **Function**: `_get_exact_candidates(db, query, limit=50)`
- **Mechanism**: Direct SQL query matching case-insensitive strings across `product_name`, `brand`, `category`, and `tags`.
- **Query Structure**:
  ```sql
  SELECT * FROM products
  WHERE lower(product_name) = :q
     OR lower(brand) = :q
     OR lower(category) = :q
     OR tags ILIKE '%' || :q || '%'
  LIMIT 50;
  ```

### Source 2: Prefix & Partial Substring Candidate Generation
- **Function**: `_get_partial_candidates(db, query, limit=50)`
- **Mechanism**: Extracts individual query tokens and constructs prefix (`token%`) and substring (`%token%`) `ILIKE` filters across product name, brand, category, tags, and description.
- **Role**: Ensures keystroke-by-keystroke responsiveness for incomplete queries (e.g., `head`, `wire`).

### Source 3: PostgreSQL `pg_trgm` Trigram Fuzzy Candidate Generation
- **Function**: `fuzzy_search_products(db, query, limit=50, min_similarity=0.01)`
- **Mechanism**: Multi-path retrieval using PostgreSQL `pg_trgm` GIN indexes:
  1. **Path 1 (Default GIN %)**: GIN trigram index pre-filter at default threshold (`0.3`). Fast path (~5–20 ms) for mild typos.
  2. **Path 2 (Lowered Threshold GIN %)**: Triggers if Path 1 yields 0 candidates. Lowers transaction-scoped `pg_trgm.similarity_threshold` to `0.1` (`SET LOCAL`) to catch moderate typos (`lptop` $\rightarrow$ `laptop`, `botle` $\rightarrow$ `bottle`).
  3. **Path 3 (Bounded Word Similarity Scan)**: Fallback sequential scan bounded by `LIMIT 200` using `strict_word_similarity()` for prefix-of-word matches.

### Source 4: Local Vector Semantic Candidate Generation
- **Function**: `semantic_search_products(db, query, limit=50, min_similarity=0.0)`
- **Mechanism**: Embeds the query text into a 384-dimensional dense vector using the local `all-MiniLM-L6-v2` ONNX model, then executes an in-memory BLAS matrix-vector dot product against all 7,500 precomputed product embeddings.
- **Top-K Extraction**: `np.argsort` extracts the top-50 product IDs in ~1.4 ms, which are then queried from PostgreSQL using `WHERE id IN (...)`.

---

## 3. Score Calculation & Normalization

Every candidate product in the union is evaluated against 4 normalized scoring functions ($0.0 \le S \le 1.0$):

### 3.1. Exact Match Score ($S_{\text{exact}}$)
Evaluates exact full string matches and token set containment:
- Full `product_name` match: **1.00**
- Exact token set equality (`set(query_tokens) == set(name_tokens)`): **0.90**
- Brand exact match: **0.85**
- Query tokens are subset of product name tokens: **0.80**
- Category exact match: **0.75**
- Tag exact match: **0.70**
- Otherwise: **0.00**

### 3.2. Prefix / Partial Score ($S_{\text{partial}}$)
Calculates token-level prefix and substring overlap across 4 product attributes:
$$S_{\text{partial}} = (0.50 \times S_{\text{name}}) + (0.25 \times S_{\text{brand}}) + (0.15 \times S_{\text{category}}) + (0.10 \times S_{\text{tags}})$$
Where each field score averages the best token match ratios:
- Token exact match: **1.00**
- Token prefix match: $0.85 \times \max(0.70, \frac{\text{len}(qt)}{\text{len}(tw)})$
- Token substring match: $0.50 \times \max(0.60, \frac{\text{len}(qt)}{\text{len}(tw)})$

### 3.3. Trigram Fuzzy Score ($S_{\text{fuzzy}}$)
Calculated via PostgreSQL `pg_trgm` field-weighted expression:
- Field weights: `product_name` (50%), `brand` (20%), `category` (15%), `tags` (15%).
- Single-token queries: $(S_{\text{token}} \times 0.75) + (\text{similarity}(q, \text{name}) \times 0.25)$
- Multi-token queries: $(\text{avg\_tok} \times 0.60) + (\text{avg\_tok} \times \text{coverage} \times 0.25) + (\text{full\_sim} \times 0.15)$

### 3.4. Dense Vector Semantic Score ($S_{\text{semantic}}$)
- Evaluated via `compute_query_similarities(query)`:
  $$S_{\text{semantic}} = \mathbf{E} \cdot \mathbf{q}$$
  where $\mathbf{E} \in \mathbb{R}^{7500 \times 384}$ is the C-contiguous embedding matrix and $\mathbf{q} \in \mathbb{R}^{384}$ is the $L_2$-normalized query vector.
- **Key Design**: Every candidate in the union map receives its true semantic cosine similarity score, even if it was retrieved via fuzzy or partial channels.

---

## 4. Combined Ranking Formula & False-Positive Suppression

### 4.1. Combined Formula
The final relevance score $S_{\text{final}} \in [0.0, 1.0]$ is computed as:

$$\boxed{S_{\text{final}} = (0.20 \times S_{\text{exact}}) + (0.15 \times S_{\text{partial}}) + (0.30 \times S_{\text{fuzzy}}) + (0.35 \times S_{\text{semantic}})}$$

| Signal | Weight | Justification |
|---|---|---|
| **Semantic ($S_{\text{semantic}}$)** | **35%** | Primary driver for intent understanding, conceptual search, and synonym matching. |
| **Fuzzy ($S_{\text{fuzzy}}$)** | **30%** | Crucial for typo tolerance, character transpositions, and phonetic misspellings. |
| **Exact ($S_{\text{exact}}$)** | **20%** | Guarantees exact matches dominate top positions without being overtaken by broad semantic matches. |
| **Partial ($S_{\text{partial}}$)** | **15%** | Provides smooth relevance boost for prefix typing and token subsets. |

### 4.2. False-Positive Filtering Rule
To prevent spurious semantic hallucinations on non-matching queries:
- **Strong Lexical Signal**: Defined as $(S_{\text{exact}} > 0) \lor (S_{\text{partial}} > 0) \lor (S_{\text{fuzzy}} \ge 0.30)$.
- **Threshold Rule**:
  $$\text{Effective Threshold} = \begin{cases} 0.10 & \text{if candidate has strong lexical signal} \\ 0.15 & \text{otherwise (pure weak semantic/fuzzy noise)} \end{cases}$$
- Candidates with $S_{\text{final}} < \text{Effective Threshold}$ are discarded.

### 4.3. Deterministic Sorting Order
Results are sorted deterministically with multi-level tie-breaking:
```python
ranked_results.sort(
    key=lambda r: (
        r.final_score,
        r.semantic_score,
        r.fuzzy_score,
        -r.product.id,
    ),
    reverse=True,
)
```

---

## 5. Ranking Walkthrough Example

### Query: `"samsng phone"` (Typo Query)

| Candidate Product | $S_{\text{exact}}$ | $S_{\text{partial}}$ | $S_{\text{fuzzy}}$ | $S_{\text{semantic}}$ | $S_{\text{final}}$ Calculation | $S_{\text{final}}$ |
|---|---|---|---|---|---|---|
| **Samsung MagSafe Compatible Phone Case** | 0.00 | 0.4250 | 0.7012 | 0.6840 | $0 + (0.15 \times 0.425) + (0.30 \times 0.701) + (0.35 \times 0.684)$ | **0.5135** |
| **Samsung Car Phone Mount Magnetic** | 0.00 | 0.4250 | 0.6850 | 0.6510 | $0 + (0.15 \times 0.425) + (0.30 \times 0.685) + (0.35 \times 0.651)$ | **0.4971** |
| **Apple iPhone 15 Pro Silicone Case** | 0.00 | 0.2500 | 0.2100 | 0.5200 | $0 + (0.15 \times 0.250) + (0.30 \times 0.210) + (0.35 \times 0.520)$ | **0.2825** |
| **Panasonic Food Processor Pro** | 0.00 | 0.0000 | 0.0200 | 0.0800 | Below 0.15 threshold | *Filtered* |

---

## 6. Latency & Offline Performance Considerations

1. **In-Memory BLAS Vector Math**: Matrix-vector product over 7,500 384-dimensional float32 vectors takes **~1.4 ms** on CPU.
2. **LRU Query Embedding Cache**: `_get_query_embedding` uses an `@lru_cache(maxsize=128)` to bypass ONNX inference for identical or repeated debounce queries.
3. **Database Round Trips**: PostgreSQL is queried only for candidate IDs and top-K hydrated rows, avoiding full-table sequential scans.
4. **Overall Measured Warm Latency**: **201.51 ms average**, with sub-100ms exact lookups.
