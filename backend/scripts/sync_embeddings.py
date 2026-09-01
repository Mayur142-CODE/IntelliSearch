"""
Standalone CLI script for Incremental Product Embedding Synchronization.

Usage:
    python scripts/sync_embeddings.py [--batch-size 64] [--force-rebuild]
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure backend root is available in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.services.embedding_sync import synchronize_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize PostgreSQL product catalog embeddings with ChromaDB."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for FastEmbed embedding generation (default: 64)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force full re-embedding of all products in PostgreSQL",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NORTHSTAR PRODUCT SEARCH — EMBEDDING SYNCHRONIZATION")
    print("=" * 60)
    print(f"Batch Size:     {args.batch_size}")
    print(f"Force Rebuild:  {args.force_rebuild}")
    print("-" * 60)

    db = SessionLocal()
    try:
        t0 = time.perf_counter()
        result = synchronize_embeddings(
            db=db,
            batch_size=args.batch_size,
            force_rebuild=args.force_rebuild,
        )
        elapsed_total = (time.perf_counter() - t0) * 1000

        print(result.summary_text())

        if result.errors:
            print(f"\n[WARNING] {len(result.errors)} errors encountered during sync:")
            for err in result.errors[:10]:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print("\n[SUCCESS] Embedding synchronization completed successfully.")
            sys.exit(0)

    except Exception as e:
        print(f"\n[ERROR] Synchronization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
