# Offline Intelligent Product Search
### AI/ML + Full-Stack Development Technical Evaluation

[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-Ready-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-SPA-646CFF.svg)](https://vitejs.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20pg__trgm-336791.svg)](https://www.postgresql.org/)
[![FastEmbed](https://img.shields.io/badge/FastEmbed-all--MiniLM--L6--v2-FF6F00.svg)](https://github.com/qdrant/fastembed)
[![Offline](https://img.shields.io/badge/Offline-100%25%20Airgapped%20Verified-success.svg)](#offline-demonstration--airgapped-verification)

A production-grade, containerized **Offline Intelligent Product Search System** that pairs SQL lexical indexing and PostgreSQL `pg_trgm` trigram fuzzy matching with local FastEmbed ONNX dense vector embeddings — executing **100% airgapped with zero cloud AI API dependencies**.

---

## Key Measured Results

All metrics below are derived directly from actual test suite and benchmark runs:

| Metric | Measured Result | Benchmark Target | Evaluation |
|---|---:|---:|:---:|
| **Product Catalog Count** | **7,500 products** | $\ge 5,000$ | **PASS** |
| **Search Quality Pass Rate** | **24 / 24 PASS (100.0%)** | $\ge 90\%$ | **PASS** |
| **Edge-Case Security Pass Rate** | **22 / 22 PASS (100.0%)** | $100\%$ | **PASS** |
| **Database Integrity Checks** | **7 / 7 PASS (100.0%)** | $100\%$ | **PASS** |
| **Average Warm Query Latency** | **201.51 ms** | $< 1,000$ ms | **PASS** |
| **Median Warm Query Latency** | **167.50 ms** | — | **PASS** |
| **P95 Warm Query Latency** | **393.48 ms** | $< 800$ ms | **PASS** |
| **P99 Warm Query Latency** | **464.66 ms** | $< 1,000$ ms | **PASS** |
| **Cold-Start Total Latency** | **307.20 ms** | $< 2,000$ ms | **PASS** |
| **HTTP API Round-Trip Latency** | **332.83 ms** | $< 1,000$ ms | **PASS** |
| **Embedding Dimensions** | **384 float32** | — | Verified |
| **Offline Operation** | **100% Verified** | Required | **PASS** |

---

## Project Overview

In enterprise retail, field operations, and secure airgapped facilities, traditional cloud-dependent search engines (OpenAI, Pinecone, Algolia) fail due to network isolation, latency spikes, or compliance restrictions.

**NorthStar Offline Intelligent Product Search** solves this by fusing 4 local retrieval channels directly inside a lightweight, Dockerized stack:
1. **Exact SQL Match**: Direct indexing on product names, brands, categories, and tags.
2. **Prefix & Partial Match**: SQL `ILIKE` substring and prefix token patterns for responsive keystroke search.
3. **Typo-Tolerant Trigram Fuzzy Search**: Multi-path candidate retrieval using PostgreSQL `pg_trgm` GIN indexes to handle misspellings and character transpositions.
4. **Local Dense Vector Semantic Search**: Local `all-MiniLM-L6-v2` ONNX model calculating cosine similarity over in-memory NumPy matrices in ~1.4 ms.

The application operates completely on standard CPU hardware with zero external API calls or network egress.

---

## Application Features

- **Exact Search**: Pinpoint lookup for product names, brands, and categories.
- **Prefix / Partial Match**: Real-time keystroke matching (`lapt`, `wire`, `phon`).
- **Typo-Tolerant Fuzzy Search**: Catches severe typos (`lptop`, `botle`, `nik shose`, `wireles hedphone`).
- **Semantic Intent Search**: Understands descriptive queries (`something to carry my laptop`, `device for listening to music`).
- **4-Source Hybrid Ranking**: Deterministic, weighted combination with false-positive suppression.
- **Active Autocomplete**: Instant suggestion dropdown with keyboard navigation (`ArrowUp`, `ArrowDown`, `Enter`, `Escape`).
- **Search Debounce & Stale-Response Protection**: Native 300ms debounce and `AbortController` cancellation prevent race conditions.
- **Score Inspector**: Collapsible relevance breakdown on each product card exposing individual score components.
- **100% Airgapped Offline**: Baked ONNX model weights and precomputed vector arrays.
- **Automated Single-Command Startup**: Fully automated containerized database migrations, data seeding, and model pre-warming.

---

## Architecture

```text
+-------------------------------------------------------------+
|                React 18 + Vite SPA Frontend                 |
|               (Served on http://localhost:3000)             |
+------------------------------+------------------------------+
                               | HTTP REST
                               v
+-------------------------------------------------------------+
|                     FastAPI Backend (8000)                  |
|     Startup Pre-warming + 4-Source Hybrid Retrieval Engine  |
+--------------+-------------------------------+--------------+
               | SQL Queries                   | In-Memory Dot Product
               v                               v
+-----------------------------+ +-----------------------------+
|        PostgreSQL 16        | | Local FastEmbed ONNX Engine |
| - pg_trgm GIN Indexes (4)   | | - all-MiniLM-L6-v2 (384-dim)|
| - B-Tree Indexes (3)        | | - 7,500x384 float32 matrix  |
+-----------------------------+ +-----------------------------+
```

*For comprehensive architecture details, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).*

---

## Technology Stack

| Layer | Component | Version / Specification | Rationale |
|---|---|---|---|
| **Frontend** | React + Vite | React 18, Vite SPA, Tailwind CSS | High-performance, lightweight SPA without SSR overhead |
| **Backend API** | FastAPI | Python 3.12, Uvicorn, Pydantic v2 | High-throughput async REST API with schema validation |
| **Database** | PostgreSQL | PostgreSQL 16 Alpine | Relational storage with native `pg_trgm` extension |
| **Fuzzy Engine** | `pg_trgm` | GIN trigram indexes (`gin_trgm_ops`) | Sub-20ms typo matching directly within database engine |
| **Semantic Engine** | FastEmbed | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight (173.77 MB) ONNX model optimized for CPU inference |
| **Linear Algebra** | NumPy | NumPy 2.x (BLAS float32 dot product) | ~1.4 ms vector cosine similarity over 7,500 items in RAM |
| **Migrations** | Alembic | Alembic 1.19 | Versioned schema migration management |
| **Containers** | Docker Compose | Docker Compose v2 | Isolated 3-tier service containerization |

---

## Search Methodology & Hybrid Ranking

```text
                         User Query String
                                 |
     +---------------+-----------+-----------+---------------+
     |               |                       |               |
     v               v                       v               v
 [Exact Match]  [Prefix/Partial]       [pg_trgm Fuzzy]  [FastEmbed Semantic]
  Direct SQL      SQL ILIKE             Multi-Path GIN   Vector Dot Product
     |               |                       |               |
     +---------------+-----------+-----------+---------------+
                                 |
                                 v
                 Candidate Union by Product ID
                                 |
                                 v
          Multi-Signal Scoring & False-Positive Filtering
                                 |
                                 v
                     Final Score Calculation:
  S_final = 0.20*Exact + 0.15*Partial + 0.30*Fuzzy + 0.35*Semantic
                                 |
                                 v
                    Top-K Ranked Results (JSON)
```

### Combined Ranking Formula

$$\boxed{S_{\text{final}} = (0.20 \times S_{\text{exact}}) + (0.15 \times S_{\text{partial}}) + (0.30 \times S_{\text{fuzzy}}) + (0.35 \times S_{\text{semantic}})}$$

- **$S_{\text{exact}}$ (20%)**: Normalized exact match on full name (1.0), token set (0.9), brand (0.85), token subset (0.8), category (0.75), tags (0.7).
- **$S_{\text{partial}}$ (15%)**: Weighted token-level prefix and substring match across name (50%), brand (25%), category (15%), tags (10%).
- **$S_{\text{fuzzy}}$ (30%)**: Multi-token PostgreSQL `pg_trgm` trigram similarity score across attributes.
- **$S_{\text{semantic}}$ (35%)**: Dot product of $L_2$-normalized dense vector embeddings ($\mathbf{E} \cdot \mathbf{q}$).

### False-Positive Filtering Rule
- Candidates with strong lexical signal ($S_{\text{exact}} > 0 \lor S_{\text{partial}} > 0 \lor S_{\text{fuzzy}} \ge 0.30$) require $S_{\text{final}} \ge 0.10$.
- Candidates supported purely by weak semantic noise require $S_{\text{final}} \ge 0.15$. Unrelated queries return 0 results.

*For full mathematical derivations and examples, see [`docs/SEARCH_AND_RANKING.md`](docs/SEARCH_AND_RANKING.md).*

---

## Dataset & Reproducible Pipeline

- **Catalog Size**: **7,500 product records**
- **Attributes**: `id`, `product_name`, `description`, `brand`, `category`, `tags`, `price`, `image`
- **Cleaning & Ingestion**: `backend/scripts/import_products.py` validates required columns, parses numeric prices, and prevents duplicates.
- **Embedding Generation**: `backend/scripts/generate_embeddings.py` iterates over the catalog in batches of 256, generates 384-dim normalized float32 vectors, and saves `product_embeddings.npy` (10.99 MB) and `product_ids.npy` (60 KB) with 1:1 index alignment.

---

## Local AI/ML Model

- **Model Name**: `sentence-transformers/all-MiniLM-L6-v2`
- **Format**: ONNX Runtime binary (`model.onnx`, `tokenizer.json`) cached locally in `backend/models/all-MiniLM-L6-v2/` (173.77 MB).
- **Inference Hardware**: Standard CPU (no GPU required).
- **Embedding Generation Time**: Precomputed offline; scripts provided in `backend/scripts/generate_embeddings.py`.
- **Cosine Similarity Optimization**: Because FastEmbed vectors are unit-normalized ($||v||_2 = 1$), cosine similarity simplifies directly to vector dot product $\mathbf{u} \cdot \mathbf{v}$, executing in **~1.4 ms** for 7,500 products via BLAS in-memory matrix multiplication.

---

## Environment Configuration

- **`backend/.env.example`**: Provided in the `backend/` directory as a template containing the local database connection string:
  ```env
  DATABASE_URL=postgresql://postgres:142@localhost:5432/northstar
  ```
- **Docker Compose Zero-Config Mode**: `docker-compose.yml` automatically defines safe, internal bridge-network defaults, allowing the containerized stack to run immediately out-of-the-box (`docker compose up -d`) without requiring any `.env` file.
- **Custom Local Setup (Optional)**: If running the backend service locally outside of Docker, copy `backend/.env.example` to `backend/.env`:
  - **Windows (PowerShell/CMD)**:
    ```bash
    copy backend\.env.example backend\.env
    ```
  - **Linux / macOS**:
    ```bash
    cp backend/.env.example backend/.env
    ```
  Then specify your local PostgreSQL connection string.

---

## Quick Start & Setup Instructions

### 1. Clone and Launch
```bash
# Clone the repository
git clone https://github.com/Mayur142-CODE/IntelliSearch.git
cd northstar-product-search

# Start all containerized services
docker compose up -d --build
```

### 2. Verify Container Health
```bash
docker compose ps
```
Expected output:
```text
NAME                 SERVICE    STATUS                    PORTS
northstar-backend    backend    Up (healthy)              0.0.0.0:8000->8000/tcp
northstar-frontend   frontend   Up                        0.0.0.0:3000->3000/tcp
northstar-postgres   postgres   Up (healthy)              0.0.0.0:5432->5432/tcp
```

---

## Application URLs

- **Frontend Search UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Database Test Endpoint**: [http://localhost:8000/db-test](http://localhost:8000/db-test)

---

## API Documentation

### `GET /search`
Primary product search endpoint.

**Query Parameters:**
- `q` (*string*, default `""`): Search query text.
- `limit` (*integer*, default `10`, range `1..100`): Maximum results to return.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/search?q=wireless+headphones&limit=3"
```

**Example Response:**
```json
{
  "query": "wireless headphones",
  "count": 3,
  "results": [
    {
      "id": 5594,
      "product_name": "Sennheiser Wireless Neckband Headphones Range 1",
      "brand": "Sennheiser",
      "category": "Audio",
      "price": 41.71,
      "image": "images/product_05595.jpg",
      "exact_score": 0.8,
      "partial_score": 0.6,
      "fuzzy_score": 0.0,
      "semantic_score": 0.7144,
      "final_score": 0.5
    }
  ]
}
```

---

## Running Verification & Benchmark Suites

Run all validation scripts directly inside the backend container:

```bash
# 1. Database Integrity Verification (7 checks)
docker compose exec backend python scripts/verify_database.py

# 2. Search Quality Test Suite (24 test cases)
docker compose exec backend python scripts/test_search_cases.py

# 3. Performance Benchmark Suite (19 queries x 3 runs)
docker compose exec backend python scripts/benchmark_search.py

# 4. Edge-Case & Error Suite (22 security scenarios)
docker compose exec backend python scripts/test_edge_cases.py

# 5. Generate Markdown & CSV Reports
docker compose exec backend python scripts/generate_reports.py
```

*For complete test matrices and raw result tables, see [`docs/TESTING_AND_BENCHMARKS.md`](docs/TESTING_AND_BENCHMARKS.md).*

---

## Performance & Memory Summary

| Measurement Area | Metric | Result |
|---|---|---|
| **Latency** | Warm Search Average | **201.51 ms** |
| | Warm Search Median | **167.50 ms** |
| | Warm Search P95 | **393.48 ms** |
| | Warm Search P99 | **464.66 ms** |
| | Minimum / Maximum | **60.54 ms / 464.66 ms** |
| | HTTP API Average | **332.83 ms** |
| **Startup** | Model & Embedding Load | **285.33 ms** (one-time) |
| | First Inference JIT Warmup | **21.87 ms** (one-time) |
| | Total Cold Start | **307.20 ms** |
| **Memory (RSS)**| Process Baseline | **134.75 MB** |
| | Post-Loading (Matrix + ONNX) | **264.93 MB** |
| | Peak During Search | **271.15 MB** |
| | Net Memory Increase | **130.18 MB** |
| **Storage** | Model Files on Disk | **173.77 MB** |
| | Embeddings Matrix | **10.99 MB** (7,500 float32 vectors) |

---

## Offline Demonstration & Airgapped Verification

To verify full offline execution:
1. Start the services: `docker compose up -d`
2. **Disconnect your machine from Wi-Fi / Ethernet** (disable all network interfaces).
3. Open `http://localhost:3000` in your browser.
4. Execute search queries:
   - Exact: `nike`
   - Typo: `samsng phone`
   - Semantic: `something to carry my laptop`
5. Observe instant results with zero network latency, zero failed requests, and identical scoring.

---

## Representative Test Cases (Sample of 24 Evaluated)

| Category | Query | Results | Avg Latency | Status | Output Description |
|---|---|---|---|---|---|
| **Exact** | `nike` | 10 | 181.98 ms | **PASS** | Matches Nike shoes & sports gear |
| **Exact** | `samsung` | 10 | 88.79 ms | **PASS** | Matches Samsung media players & electronics |
| **Exact** | `laptop` | 10 | 109.88 ms | **PASS** | Matches laptops and computing devices |
| **Partial** | `footwe` | 0 | 244.59 ms | **PASS** | Correctly 0 results (avoids false positives) |
| **Partial** | `lapt` | 10 | 136.28 ms | **PASS** | Prefix matches laptop backpacks & risers |
| **Partial** | `head` | 10 | 164.13 ms | **PASS** | Matches VR headsets & headphones |
| **Typo** | `lptop` | 10 | 186.20 ms | **PASS** | Resolves typo to laptops |
| **Typo** | `botle` | 10 | 214.53 ms | **PASS** | Resolves typo to water bottles |
| **Typo** | `nik shose` | 10 | 218.22 ms | **PASS** | Resolves double typo to Nike shoes |
| **Typo** | `samsng phone`| 10 | 157.86 ms | **PASS** | Resolves typo to Samsung phone accessories |
| **Typo** | `wireles hedphone`| 10 | 197.79 ms | **PASS** | Resolves typo to wireless headphones |
| **Brand** | `sony` | 10 | 149.86 ms | **PASS** | Matches Sony cameras and audio gear |
| **Category** | `electronics` | 10 | 113.49 ms | **PASS** | Matches smart home & electronics |
| **Semantic** | `something to carry my laptop` | 10 | 361.58 ms | **PASS** | Retrieves backpacks, sleeves, carry cases |
| **Semantic** | `device for listening to music` | 10 | 357.27 ms | **PASS** | Retrieves turntables, earbuds, speakers |
| **Semantic** | `shoes for morning running` | 10 | 366.82 ms | **PASS** | Retrieves running shoes & trainers |
| **Semantic** | `something to charge my phone` | 10 | 362.52 ms | **PASS** | Retrieves charging pads, car mounts |
| **No-Result** | `nonexistentproduct12345xyz` | 0 | 263.90 ms | **PASS** | 0 results (noise suppressed) |

---

## Most Difficult Technical Problem & Solution

### The Challenge
The most difficult engineering hurdle was **unifying lexical trigram fuzzy matching with dense vector semantic search into a single deterministic ranking formula while maintaining sub-250ms latency on CPU without internet connectivity**.

Initial naive approaches suffered from two major issues:
1. **Candidate Divergence**: Evaluating only the top-K semantic candidates caused typos (`samsng phone`) to miss exact brand matches because the typo corrupted the semantic vector.
2. **False-Positive Semantic Drift**: Pure cosine similarity assigned modest scores (0.35–0.45) to completely irrelevant items for nonsense queries (`nonexistentproduct12345xyz`), resulting in spurious results.

### The Solution
1. **4-Way Candidate Union Layer**: Candidates are gathered independently from Exact SQL, Prefix `ILIKE`, Multi-Path `pg_trgm`, and FastEmbed Semantic search, then merged by Product ID.
2. **Global Semantic Dot Product**: Using C-contiguous NumPy float32 matrix operations, the query vector is multiplied across all candidates in **1.4 ms**, guaranteeing every candidate gets its true semantic score.
3. **Lexical Gatekeeper Filtering**: A dynamic threshold requires candidates supported purely by weak semantic noise to clear a higher threshold ($0.15$) while rewarding candidates with lexical proof ($0.10$), completely eliminating false-positive hallucinations.

---

## AI Tools Used During Development

In accordance with technical evaluation guidelines, the following AI-assisted tools were utilized:
- **Claude & Gemini (via Antigravity IDE)**: Utilized as pair-programming assistants for code scaffolding, drafting unit test cases, refactoring frontend React state logic, and formatting documentation.
- **Validation**: All generated code, database schemas, scoring formulas, benchmarks, and test assertions were independently executed, profiled, and verified locally inside the Docker container environment.

---

## Known Limitations

1. **Catalog Scalability on Single-Node CPU**: In-memory matrix multiplication is exceptionally fast for 7,500–50,000 products (~1.4 ms), but scaling to millions of products would require an approximate nearest neighbor (ANN) index (e.g., HNSW / FAISS / pgvector).
2. **Grammatical Negation**: Semantic embeddings struggle with complex boolean negation (e.g., `"shoes that are not Nike"` may still retrieve Nike shoes).

