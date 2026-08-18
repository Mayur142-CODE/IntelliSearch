"""
Database Verification Script for Offline Intelligent Product Search.

Reports:
- Database connection status
- Actual product count (queried, not hardcoded)
- Table names in public schema
- Key indexes on the products table
- Embedding file alignment check

Does NOT expose passwords or secrets.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sqlalchemy import inspect, text
from app.core.database import SessionLocal, engine
from app.models.product import Product

RESULTS_DIR = BACKEND_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "database_verification.json"


def run_verification():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [],
    }

    def add_check(name, status, detail):
        report["checks"].append({"check": name, "status": status, "detail": detail})
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}: {detail}")

    print("=" * 60)
    print("DATABASE VERIFICATION REPORT")
    print("=" * 60)

    # 1. Connection
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        add_check("Database Connection", "PASS", "PostgreSQL connection successful")
    except Exception as e:
        add_check("Database Connection", "FAIL", str(e))
        report["overall"] = "FAIL"
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {RESULTS_FILE}")
        return report

    # 2. Product count (queried, not hardcoded)
    try:
        product_count = db.query(Product).count()
        populated = product_count > 0
        add_check(
            "Product Count",
            "PASS" if populated else "FAIL",
            f"{product_count} products in database"
            + ("" if populated else " (database appears empty)"),
        )
        report["product_count"] = product_count
    except Exception as e:
        add_check("Product Count", "FAIL", str(e))

    # 3. Table names
    try:
        inspector = inspect(engine)
        tables = sorted(inspector.get_table_names(schema="public"))
        has_products = "products" in tables
        add_check(
            "Products Table Exists",
            "PASS" if has_products else "FAIL",
            f"Tables found: {tables}",
        )
        report["tables"] = tables
    except Exception as e:
        add_check("Products Table Exists", "FAIL", str(e))

    # 4. Key indexes
    try:
        indexes = inspector.get_indexes("products", schema="public")
        index_names = [idx["name"] for idx in indexes]
        add_check(
            "Product Indexes",
            "PASS" if len(index_names) > 0 else "FAIL",
            f"{len(index_names)} indexes: {index_names}",
        )
        report["indexes"] = index_names
    except Exception as e:
        add_check("Product Indexes", "FAIL", str(e))

    # 5. Column schema
    try:
        columns = inspector.get_columns("products", schema="public")
        col_names = [c["name"] for c in columns]
        expected = {"id", "product_name", "description", "brand", "category", "tags", "price", "image"}
        missing = expected - set(col_names)
        add_check(
            "Column Schema",
            "PASS" if not missing else "FAIL",
            f"Columns: {col_names}" + (f" (missing: {missing})" if missing else ""),
        )
        report["columns"] = col_names
    except Exception as e:
        add_check("Column Schema", "FAIL", str(e))

    # 6. Embedding files alignment
    try:
        import numpy as np
        emb_file = BACKEND_DIR / "data" / "embeddings" / "product_embeddings.npy"
        ids_file = BACKEND_DIR / "data" / "embeddings" / "product_ids.npy"

        if emb_file.exists() and ids_file.exists():
            emb = np.load(emb_file)
            ids = np.load(ids_file)
            emb_count = emb.shape[0]
            ids_count = ids.shape[0]
            aligned = emb_count == ids_count
            matches_db = emb_count == product_count if 'product_count' in report else False
            add_check(
                "Embedding Alignment",
                "PASS" if aligned and matches_db else "FAIL",
                f"Embeddings: {emb_count}, Product IDs: {ids_count}, DB: {report.get('product_count', '?')}"
                + (" (aligned)" if aligned and matches_db else " (MISMATCH)"),
            )
        else:
            add_check("Embedding Alignment", "FAIL", "Embedding files not found")
    except Exception as e:
        add_check("Embedding Alignment", "FAIL", str(e))

    # 7. pg_trgm extension
    try:
        result = db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'pg_trgm'")).fetchone()
        add_check(
            "pg_trgm Extension",
            "PASS" if result else "FAIL",
            "pg_trgm extension is installed" if result else "pg_trgm extension NOT found",
        )
    except Exception as e:
        add_check("pg_trgm Extension", "FAIL", str(e))

    db.close()

    # Overall
    all_pass = all(c["status"] == "PASS" for c in report["checks"])
    report["overall"] = "PASS" if all_pass else "FAIL"

    print(f"\nOverall: {report['overall']}")
    print("=" * 60)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

    return report


if __name__ == "__main__":
    run_verification()
