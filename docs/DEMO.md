# Demonstration & Video Script Guide

This document provides the structured guide and timeline for recording the 3–5 minute demonstration video for the **NorthStar Offline Intelligent Product Search** technical evaluation.

---

## Demonstration Video Timeline (3–5 Minutes)

```
00:00 - 00:30  |  1. Project Introduction & Architecture
00:30 - 01:00  |  2. Docker Environment & Container Health
01:00 - 01:45  |  3. Exact & Prefix / Partial Search
01:45 - 02:30  |  4. Typo-Tolerant Trigram Fuzzy Search
02:30 - 03:15  |  5. Semantic Vector Search & Intent Understanding
03:15 - 03:45  |  6. Relevance Score Breakdown & Autocomplete
03:45 - 04:30  |  7. Offline Airgapped Verification (Disconnected Network)
04:30 - 05:00  |  8. Performance Benchmarks & Conclusion
```

---

## Step-by-Step Demonstration Walkthrough

### 1. Introduction & Overview (00:00 – 00:30)
- **Visual**: Web browser open at `http://localhost:3000`.
- **Narration**: Introduce the NorthStar Offline Intelligent Product Search system — an airgapped, high-performance product discovery application combining SQL exact matching, PostgreSQL `pg_trgm` trigram fuzzy search, and local FastEmbed ONNX dense vector embeddings with zero cloud API dependencies.

### 2. Docker Containers & Status (00:30 – 01:00)
- **Action**: Switch to terminal and run `docker compose ps`.
- **Narration**: Show all 3 services (`northstar-frontend` on 3000, `northstar-backend` on 8000, `northstar-postgres` on 5432) running in healthy state. Highlight that the backend entrypoint automatically executed database readiness, Alembic migrations, and idempotent product seeding.

### 3. Exact & Prefix Search (01:00 – 01:45)
- **Action 1**: Type `nike` in the search bar.
  - *Observation*: Instant top results from Nike, including running shoes and sports accessories.
- **Action 2**: Type `lapt`.
  - *Observation*: Instant prefix matching for laptops, laptop stands, and laptop backpacks.
- **Action 3**: Type `wire`.
  - *Observation*: Matches wireless earbuds, headphones, and chargers.

### 4. Typo-Tolerant Fuzzy Search (01:45 – 02:30)
- **Action 1**: Type `lptop` (missing 'a').
  - *Observation*: Resolves directly to laptop computers and accessories.
- **Action 2**: Type `botle` (missing 't').
  - *Observation*: Resolves directly to insulated water bottles and bottle sets.
- **Action 3**: Type `nik shose` (double typo).
  - *Observation*: Correctly matches Nike footwear and sneakers.
- **Action 4**: Type `wireles hedphone` (double typo).
  - *Observation*: Correctly retrieves wireless noise-cancelling headphones.

### 5. Semantic Vector Search (02:30 – 03:15)
- **Action 1**: Type `something to carry my laptop`.
  - *Observation*: Retrieves laptop backpacks, sleeves, and carry cases (even when the title doesn't contain all search terms).
- **Action 2**: Type `device for listening to music`.
  - *Observation*: Retrieves turntables, wireless earbuds, and portable speakers.
- **Action 3**: Type `shoes for morning running`.
  - *Observation*: Retrieves cushioned running shoes and trainers across brands.

### 6. Autocomplete & Score Inspector (03:15 – 03:45)
- **Action 1**: Highlight the instant autocomplete suggestion dropdown and navigate using `ArrowDown` / `ArrowUp` keys.
- **Action 2**: Click **"Relevance Breakdown"** on a product card.
  - *Observation*: Display the exact values for Final Score, Exact Score, Partial Score, Fuzzy Score, and Semantic Score.

### 7. Airgapped Offline Verification (03:45 – 04:30)
- **Action 1**: Disconnect the host machine from Wi-Fi / Ethernet or disable network adapter.
- **Action 2**: Perform queries (`samsng phone`, `bag for traveling`, `device for listening to music`).
- **Observation**: All searches execute instantly with identical latency and scoring. Zero network requests fail because model weights, embeddings, and database run 100% locally.

### 8. Performance Benchmark Summary (04:30 – 05:00)
- **Visual**: Show `backend/results/performance_report.md` or Swagger UI at `http://localhost:8000/docs`.
- **Narration**: Conclude with measured metrics: **7,500 products**, **24/24 quality pass rate**, **201.51 ms average warm latency**, and **307.2 ms cold start**.
