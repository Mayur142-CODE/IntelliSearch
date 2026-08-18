# Performance Benchmark Report

**Generated from:** `benchmark_results.json`

## Dataset

| Metric | Value |
|---|---|
| Products | 7500 |
| Embeddings | 7500 |
| Dimensions | 384 |
| Model | all-MiniLM-L6-v2 |

## Memory Footprint

| Metric | Value |
|---|---|
| Baseline | 134.75 MB |
| Post-Load | 264.93 MB |
| Peak | 271.15 MB |
| Net Increase | 130.18 MB |

## Cold-Start Latency

| Metric | Value |
|---|---|
| Model + Embeddings Load | 285.33 ms |
| First ONNX Inference | 21.87 ms |
| Total Cold Start | 307.2 ms |

## Warm Search Latency

| Metric | Value |
|---|---|
| Queries Tested | 19 |
| Average | 201.51 ms |
| Median | 167.5 ms |
| P95 | 393.48 ms |
| P99 | 464.66 ms |
| Minimum | 60.54 ms |
| Maximum | 464.66 ms |

## Per-Query Breakdown

| Query | Type | Avg (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| `nike` | Exact | 140.13 | 132.44 | 146.54 |
| `samsung` | Exact | 65.36 | 60.54 | 74.98 |
| `laptop` | Partial | 129.54 | 106.61 | 143.07 |
| `head` | Partial | 130.51 | 118.51 | 150.76 |
| `lapt` | Partial | 84.84 | 84.25 | 85.36 |
| `wire` | Partial | 116.03 | 104.65 | 121.98 |
| `foot` | Partial | 172.93 | 163.49 | 187.81 |
| `lptop` | Fuzzy | 156.53 | 155.53 | 158.03 |
| `botle` | Fuzzy | 194.3 | 188.76 | 199.1 |
| `footwe` | Fuzzy | 270.13 | 243.21 | 294.47 |
| `nik shose` | Fuzzy | 212.33 | 205.78 | 225.16 |
| `samsng phone` | Fuzzy | 147.41 | 143.51 | 152.51 |
| `wireles hedphone` | Fuzzy | 232.54 | 187.31 | 297.22 |
| `something to carry my laptop` | Semantic | 322.17 | 306.11 | 342.4 |
| `device for listening to music` | Semantic | 428.7 | 393.48 | 464.66 |
| `shoes for morning running` | Semantic | 334.78 | 318.4 | 358.63 |
| `something to charge my phone` | Semantic | 292.41 | 282.66 | 302.05 |
| `bag for traveling` | Semantic | 143.53 | 133.58 | 153.91 |
| `nonexistentproduct12345xyz` | No Result | 254.59 | 246.1 | 262.97 |

## Per-Type Aggregation

| Type | Avg (ms) | Median (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Exact | 102.75 | 103.71 | 60.54 | 146.54 |
| Partial | 126.77 | 121.98 | 84.25 | 187.81 |
| Fuzzy | 202.21 | 197.07 | 143.51 | 297.22 |
| Semantic | 304.32 | 318.02 | 133.58 | 464.66 |
| No Result | 254.59 | 254.69 | 246.1 | 262.97 |

## HTTP API Latency

| Metric | Value |
|---|---|
| Mode | Live HTTP API Server (http://localhost:8000) |
| Average | 332.83 ms |

## Performance Conclusion

With an average warm search latency of **201.51 ms**, performance is **very good**.
