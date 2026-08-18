"""
Report Generation Script for Offline Intelligent Product Search.

Parses existing JSON results from:
  - backend/search_test_results.json (search quality)
  - backend/benchmark_results.json (performance)
  - backend/results/edge_case_results.json (edge cases)
  - backend/results/database_verification.json (database)

Generates human-readable Markdown + CSV reports under backend/results/.

IMPORTANT: This script never invents data. It only reads and formats
actual measured results from JSON files.
"""

import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, median

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BACKEND_DIR / "results"


def generate_search_quality_report():
    """Generate search quality report from test_search_cases.py output."""
    src = BACKEND_DIR / "search_test_results.json"
    if not src.exists():
        print(f"[SKIP] {src} not found — run test_search_cases.py first.")
        return

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("individual_test_results", [])
    total = data.get("total_tests", len(tests))
    passed = data.get("passed", 0)
    failed = data.get("failed", 0)
    pass_rate = data.get("pass_rate", "?")
    latency = data.get("overall_latency", {})

    # Per-type breakdown
    type_stats = {}
    for t in tests:
        st = t["search_type"]
        if st not in type_stats:
            type_stats[st] = {"total": 0, "passed": 0, "failed": 0}
        type_stats[st]["total"] += 1
        if t["status"] == "PASS":
            type_stats[st]["passed"] += 1
        else:
            type_stats[st]["failed"] += 1

    # Identify failures and successes
    failures = [t for t in tests if t["status"] == "FAIL"]
    successes = [t for t in tests if t["status"] == "PASS"]

    # Write MD report
    md_path = RESULTS_DIR / "search_quality_report.md"
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# Search Quality Report\n\n")
        md.write(f"**Generated from:** `search_test_results.json`\n\n")
        md.write(f"## Summary\n\n")
        md.write(f"| Metric | Value |\n|---|---|\n")
        md.write(f"| Total Tests | {total} |\n")
        md.write(f"| Passed | {passed} |\n")
        md.write(f"| Failed | {failed} |\n")
        md.write(f"| Pass Rate | {pass_rate} |\n")
        md.write(f"| Average Latency | {latency.get('avg_ms', '?')} ms |\n")
        md.write(f"| Median Latency | {latency.get('median_ms', '?')} ms |\n")
        md.write(f"| Max Latency | {latency.get('max_ms', '?')} ms |\n")

        md.write(f"\n## Per-Category Breakdown\n\n")
        md.write(f"| Category | Total | Passed | Failed | Pass Rate |\n|---|---|---|---|---|\n")
        for cat, stats in type_stats.items():
            rate = f"{(stats['passed']/stats['total'])*100:.0f}%" if stats['total'] > 0 else "N/A"
            md.write(f"| {cat} | {stats['total']} | {stats['passed']} | {stats['failed']} | {rate} |\n")

        # Example successes
        md.write(f"\n## Successful Search Examples\n\n")
        for t in successes[:5]:
            md.write(f"### Query: `{t['query']}` ({t['search_type']})\n")
            md.write(f"- **Status:** PASS\n")
            md.write(f"- **Results:** {t['num_results']}\n")
            md.write(f"- **Avg Latency:** {t['latency_ms']['avg']} ms\n")
            if t["top_5_product_names"]:
                md.write(f"- **Top Results:** {', '.join(t['top_5_product_names'][:3])}\n")
            md.write(f"- **Explanation:** {t['explanation']}\n\n")

        # Failures
        if failures:
            md.write(f"\n## Failed Tests\n\n")
            for t in failures:
                md.write(f"### Query: `{t['query']}` ({t['search_type']})\n")
                md.write(f"- **Status:** FAIL\n")
                md.write(f"- **Results:** {t['num_results']}\n")
                md.write(f"- **Failure Reason:** {t['explanation']}\n")
                if t["top_5_product_names"]:
                    md.write(f"- **Top Results:** {', '.join(t['top_5_product_names'][:3])}\n")
                md.write(f"\n")
        else:
            md.write(f"\n## Failed Tests\n\nNone — all {total} tests passed.\n")

        md.write(f"\n## Conclusion\n\n")
        md.write(f"The search quality evaluation achieved a **{pass_rate}** pass rate ")
        md.write(f"across {total} test cases covering exact, prefix/partial, typo/fuzzy, ")
        md.write(f"brand, category, semantic, and no-result queries.\n")

    print(f"  [OK] {md_path}")

    # Write CSV
    csv_path = RESULTS_DIR / "search_quality_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["test_number", "query", "search_type", "status", "num_results",
                         "avg_latency_ms", "explanation"])
        for t in tests:
            writer.writerow([
                t["test_number"], t["query"], t["search_type"], t["status"],
                t["num_results"], t["latency_ms"]["avg"], t["explanation"]
            ])
    print(f"  [OK] {csv_path}")


def generate_performance_report():
    """Generate performance benchmark report from benchmark_search.py output."""
    src = BACKEND_DIR / "benchmark_results.json"
    if not src.exists():
        print(f"[SKIP] {src} not found — run benchmark_search.py first.")
        return

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    perf = data.get("warm_search_performance", {})
    queries = perf.get("queries", [])
    dataset = data.get("dataset", {})
    model = data.get("model", {})
    memory = data.get("memory", {})
    cold = data.get("cold_start", {})
    api_perf = data.get("api_performance", {})

    # Compute P95/P99 from all raw runs
    all_latencies = []
    for q in queries:
        all_latencies.extend(q.get("raw_runs_ms", [q.get("avg_ms", 0)]))

    all_latencies.sort()
    n = len(all_latencies)
    p95 = all_latencies[int(n * 0.95)] if n > 0 else 0
    p99 = all_latencies[int(n * 0.99)] if n > 0 else 0

    md_path = RESULTS_DIR / "performance_report.md"
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# Performance Benchmark Report\n\n")
        md.write(f"**Generated from:** `benchmark_results.json`\n\n")

        md.write("## Dataset\n\n")
        md.write(f"| Metric | Value |\n|---|---|\n")
        md.write(f"| Products | {dataset.get('products_count', '?')} |\n")
        md.write(f"| Embeddings | {dataset.get('embeddings_count', '?')} |\n")
        md.write(f"| Dimensions | {dataset.get('embedding_dimensions', '?')} |\n")
        md.write(f"| Model | {model.get('name', '?')} |\n")

        md.write(f"\n## Memory Footprint\n\n")
        md.write(f"| Metric | Value |\n|---|---|\n")
        md.write(f"| Baseline | {memory.get('baseline_mb', '?')} MB |\n")
        md.write(f"| Post-Load | {memory.get('post_loading_mb', '?')} MB |\n")
        md.write(f"| Peak | {memory.get('peak_mb', '?')} MB |\n")
        md.write(f"| Net Increase | {memory.get('net_increase_mb', '?')} MB |\n")

        md.write(f"\n## Cold-Start Latency\n\n")
        md.write(f"| Metric | Value |\n|---|---|\n")
        md.write(f"| Model + Embeddings Load | {cold.get('model_embedding_load_ms', '?')} ms |\n")
        md.write(f"| First ONNX Inference | {cold.get('first_onnx_inference_ms', '?')} ms |\n")
        md.write(f"| Total Cold Start | {cold.get('total_cold_start_ms', '?')} ms |\n")

        md.write(f"\n## Warm Search Latency\n\n")
        md.write(f"| Metric | Value |\n|---|---|\n")
        md.write(f"| Queries Tested | {len(queries)} |\n")
        md.write(f"| Average | {perf.get('overall_avg_ms', '?')} ms |\n")
        md.write(f"| Median | {perf.get('overall_median_ms', '?')} ms |\n")
        md.write(f"| P95 | {round(p95, 2)} ms |\n")
        md.write(f"| P99 | {round(p99, 2)} ms |\n")
        md.write(f"| Minimum | {perf.get('overall_min_ms', '?')} ms |\n")
        md.write(f"| Maximum | {perf.get('overall_max_ms', '?')} ms |\n")

        md.write(f"\n## Per-Query Breakdown\n\n")
        md.write(f"| Query | Type | Avg (ms) | Min (ms) | Max (ms) |\n|---|---|---|---|---|\n")
        for q in queries:
            md.write(f"| `{q['query']}` | {q['type']} | {q['avg_ms']} | {q['min_ms']} | {q['max_ms']} |\n")

        type_summary = perf.get("per_type_summary", {})
        if type_summary:
            md.write(f"\n## Per-Type Aggregation\n\n")
            md.write(f"| Type | Avg (ms) | Median (ms) | Min (ms) | Max (ms) |\n|---|---|---|---|---|\n")
            for qt, stats in type_summary.items():
                md.write(f"| {qt} | {stats['avg_ms']} | {stats['median_ms']} | {stats['min_ms']} | {stats['max_ms']} |\n")

        # HTTP API
        if api_perf.get("avg_api_response_ms"):
            md.write(f"\n## HTTP API Latency\n\n")
            md.write(f"| Metric | Value |\n|---|---|\n")
            md.write(f"| Mode | {api_perf.get('mode', '?')} |\n")
            md.write(f"| Average | {api_perf.get('avg_api_response_ms', '?')} ms |\n")

        md.write(f"\n## Performance Conclusion\n\n")
        avg = perf.get("overall_avg_ms", 9999)
        if isinstance(avg, (int, float)):
            if avg < 100:
                md.write(f"With an average warm search latency of **{avg} ms**, performance is **excellent**.\n")
            elif avg < 500:
                md.write(f"With an average warm search latency of **{avg} ms**, performance is **very good**.\n")
            elif avg < 1000:
                md.write(f"With an average warm search latency of **{avg} ms**, performance is **acceptable**.\n")
            else:
                md.write(f"With an average warm search latency of **{avg} ms**, performance **needs optimization**.\n")

    print(f"  [OK] {md_path}")

    # CSV
    csv_path = RESULTS_DIR / "performance_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["query", "type", "avg_ms", "min_ms", "max_ms", "median_ms"])
        for q in queries:
            writer.writerow([q["query"], q["type"], q["avg_ms"], q["min_ms"], q["max_ms"], q["median_ms"]])
    print(f"  [OK] {csv_path}")


def generate_edge_case_report():
    """Generate edge-case test report from test_edge_cases.py output."""
    src = RESULTS_DIR / "edge_case_results.json"
    if not src.exists():
        print(f"[SKIP] {src} not found — run test_edge_cases.py first.")
        return

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    total = data.get("total_tests", len(tests))
    passed = data.get("passed", 0)
    failed = total - passed

    md_path = RESULTS_DIR / "edge_case_report.md"
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# Edge-Case & Error Testing Report\n\n")
        md.write(f"**Generated from:** `edge_case_results.json`\n\n")
        md.write(f"## Summary\n\n")
        md.write(f"| Metric | Value |\n|---|---|\n")
        md.write(f"| Total Tests | {total} |\n")
        md.write(f"| Passed | {passed} |\n")
        md.write(f"| Failed | {failed} |\n")
        md.write(f"| Pass Rate | {data.get('pass_rate', '?')} |\n")

        md.write(f"\n## Test Results\n\n")
        md.write(f"| # | Description | Status | Expected | Actual |\n|---|---|---|---|---|\n")
        for t in tests:
            status_icon = "✓" if t["status"] == "PASS" else "✗"
            md.write(f"| {t['test_id']} | {t['description']} | {status_icon} {t['status']} | {t['expected_behavior']} | {t['actual_behavior']} |\n")

        if failed > 0:
            md.write(f"\n## Failed Tests Detail\n\n")
            for t in tests:
                if t["status"] == "FAIL":
                    md.write(f"### Test {t['test_id']}: {t['description']}\n")
                    md.write(f"- **Input:** `{t['input']}`\n")
                    md.write(f"- **Expected:** {t['expected_behavior']}\n")
                    md.write(f"- **Actual:** {t['actual_behavior']}\n\n")

    print(f"  [OK] {md_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING REPORTS FROM TEST RESULTS")
    print("=" * 60)

    print("\n[1/3] Search Quality Report...")
    generate_search_quality_report()

    print("\n[2/3] Performance Report...")
    generate_performance_report()

    print("\n[3/3] Edge-Case Report...")
    generate_edge_case_report()

    print("\nAll reports generated successfully.")


if __name__ == "__main__":
    main()
