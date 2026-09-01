"""
Comprehensive Automated Test Suite for Incremental Product Embedding Synchronization.

Tests:
1. Initial Sync & Legacy Hash Backfill (Preserves existing vectors)
2. Idempotency (0 new, 0 updated, 0 deleted on rerun)
3. New Product Addition (Creates 1 embedding, searchable semantically)
4. Product Modification (Re-embeds only 1 modified product)
5. Product Deletion (Removes vector from ChromaDB)
6. Batch Processing (Inserts multiple products and verifies batch embedding)
7. Non-Blocking Concurrency Guard (Locking behavior)
"""

from decimal import Decimal
import sys
import time
from pathlib import Path

# Ensure backend root is available in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import chromadb
from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.product import Product
from app.services.embedding_sync import (
    COLLECTION_NAME,
    CHROMA_DIR,
    build_product_text,
    calculate_product_hash,
    get_chroma_collection,
    synchronize_embeddings,
)
from app.services.semantic_search import semantic_search_products


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def run(self, name: str, fn):
        print(f"\n[RUNNING] {name}...")
        t0 = time.perf_counter()
        try:
            fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"  [PASS] {name} ({elapsed_ms:.1f}ms)")
            self.passed += 1
            self.tests.append((name, True, elapsed_ms, ""))
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"  [FAIL] {name} ({elapsed_ms:.1f}ms): {e}")
            import traceback
            traceback.print_exc()
            self.failed += 1
            self.tests.append((name, False, elapsed_ms, str(e)))


def main():
    runner = TestRunner()
    db = SessionLocal()
    collection = get_chroma_collection(create_if_missing=True)

    print("=" * 80)
    print("INCREMENTAL EMBEDDING SYNCHRONIZATION TEST SUITE")
    print("=" * 80)

    try:
        # -------------------------------------------------------------
        # Test 1: Initial Sync & Legacy Metadata Backfill
        # -------------------------------------------------------------
        def test_initial_sync_and_backfill():
            res = synchronize_embeddings(db, batch_size=64)
            assert res.pg_count > 0, "Expected PostgreSQL products > 0"
            assert res.failed_count == 0, f"Expected 0 failures, got {res.failed_count}"
            assert res.deleted_count == 0, f"Expected 0 deletions, got {res.deleted_count}"
            print(f"    Initial sync: {res.pg_count} products in PG, {res.backfilled_hash_count} hashes backfilled, {res.new_count} new embeds")

        runner.run("1. Initial Sync & Metadata Backfill", test_initial_sync_and_backfill)

        # -------------------------------------------------------------
        # Test 2: Idempotency (0 changes on rerun)
        # -------------------------------------------------------------
        def test_idempotency():
            res = synchronize_embeddings(db, batch_size=64)
            assert res.new_count == 0, f"Expected 0 new embeddings, got {res.new_count}"
            assert res.updated_count == 0, f"Expected 0 updated embeddings, got {res.updated_count}"
            assert res.deleted_count == 0, f"Expected 0 deleted vectors, got {res.deleted_count}"
            assert res.backfilled_hash_count == 0, f"Expected 0 backfilled hashes, got {res.backfilled_hash_count}"
            assert res.unchanged_count == res.pg_count, f"Expected {res.pg_count} unchanged, got {res.unchanged_count}"
            assert res.has_changes is False, "Expected has_changes to be False"
            print(f"    Idempotency verified: {res.unchanged_count} products skipped in {res.total_time_ms:.1f}ms")

        runner.run("2. Idempotency on Repeated Sync", test_idempotency)

        # -------------------------------------------------------------
        # Test 3: New Product Insertion & Semantic Search Retrieval
        # -------------------------------------------------------------
        test_product_id = None

        def test_new_product_insertion():
            nonlocal test_product_id
            test_prod = Product(
                product_name="SyncTest Aurora Pro Mechanical Gaming Keyboard",
                description="Ultra-responsive mechanical keyboard with customizable RGB backlighting, tactile blue switches, and aluminum chassis.",
                brand="SyncTestBrand",
                category="Computers & Accessories",
                tags="keyboard, mechanical, gaming, rgb, synctest",
                price=Decimal("149.99"),
                image="https://example.com/synctest-keyboard.jpg",
            )
            db.add(test_prod)
            db.commit()
            db.refresh(test_prod)
            test_product_id = test_prod.id
            assert test_product_id is not None

            # Run sync
            res = synchronize_embeddings(db, batch_size=64)
            assert res.new_count == 1, f"Expected exactly 1 new embedding, got {res.new_count}"
            assert res.updated_count == 0, f"Expected 0 updated, got {res.updated_count}"
            assert res.deleted_count == 0, f"Expected 0 deleted, got {res.deleted_count}"

            # Verify in ChromaDB
            chroma_item = collection.get(ids=[str(test_product_id)], include=["metadatas", "documents"])
            assert len(chroma_item["ids"]) == 1, "Product vector missing from ChromaDB"
            meta = chroma_item["metadatas"][0]
            assert meta["product_name"] == "SyncTest Aurora Pro Mechanical Gaming Keyboard"
            assert meta["brand"] == "SyncTestBrand"
            assert meta["embedding_hash"] is not None
            assert len(meta["embedding_hash"]) == 64

            # Verify semantic search finds this new product
            search_res = semantic_search_products(db, "SyncTest Aurora Pro Mechanical Gaming Keyboard", limit=5)
            found = any(r.product.id == test_product_id for r in search_res)
            assert found, f"New product ID {test_product_id} not retrieved in semantic search"
            print(f"    New product ID {test_product_id} inserted, embedded, and verified via semantic search.")

        runner.run("3. New Product Insertion & Semantic Search Integration", test_new_product_insertion)

        # -------------------------------------------------------------
        # Test 4: Product Content Update & Re-Embedding
        # -------------------------------------------------------------
        def test_product_modification():
            assert test_product_id is not None
            prod = db.get(Product, test_product_id)
            assert prod is not None

            # Modify product content significantly
            prod.product_name = "SyncTest Quantum Stealth Silent Wireless Mouse"
            prod.description = "Ergonomic ultra-lightweight silent wireless gaming mouse with 26000 DPI optical sensor and optical switches."
            prod.tags = "mouse, wireless, gaming, silent, synctest"
            db.commit()

            # Run sync
            res = synchronize_embeddings(db, batch_size=64)
            assert res.new_count == 0, f"Expected 0 new embeddings, got {res.new_count}"
            assert res.updated_count == 1, f"Expected exactly 1 updated embedding, got {res.updated_count}"
            assert res.deleted_count == 0, f"Expected 0 deleted, got {res.deleted_count}"

            # Verify updated metadata in ChromaDB
            chroma_item = collection.get(ids=[str(test_product_id)], include=["metadatas", "documents"])
            meta = chroma_item["metadatas"][0]
            assert meta["product_name"] == "SyncTest Quantum Stealth Silent Wireless Mouse"

            # Verify semantic search retrieves updated query
            search_res = semantic_search_products(db, "SyncTest Quantum Stealth Silent Wireless Mouse", limit=5)
            found = any(r.product.id == test_product_id for r in search_res)
            assert found, f"Updated product ID {test_product_id} not retrieved for modified mouse query"
            print(f"    Modified product ID {test_product_id} re-embedded and verified via semantic search.")

        runner.run("4. Product Content Update & Selective Re-Embedding", test_product_modification)

        # -------------------------------------------------------------
        # Test 5: Product Deletion & Vector Cleanup
        # -------------------------------------------------------------
        def test_product_deletion():
            assert test_product_id is not None
            prod = db.get(Product, test_product_id)
            assert prod is not None

            # Delete from PostgreSQL
            db.delete(prod)
            db.commit()

            # Run sync
            res = synchronize_embeddings(db, batch_size=64)
            assert res.new_count == 0, f"Expected 0 new embeddings, got {res.new_count}"
            assert res.updated_count == 0, f"Expected 0 updated embeddings, got {res.updated_count}"
            assert res.deleted_count == 1, f"Expected exactly 1 deleted vector, got {res.deleted_count}"

            # Verify removed from ChromaDB
            chroma_item = collection.get(ids=[str(test_product_id)], include=["metadatas"])
            assert len(chroma_item["ids"]) == 0, f"Stale vector for product {test_product_id} still in ChromaDB"
            print(f"    Deleted product ID {test_product_id} removed from ChromaDB vectors.")

        runner.run("5. Product Deletion & Stale Vector Removal", test_product_deletion)

        # -------------------------------------------------------------
        # Test 6: Batch Processing Multiple Additions
        # -------------------------------------------------------------
        batch_ids = []

        def test_batch_processing():
            nonlocal batch_ids
            # Insert 4 test products
            for i in range(1, 5):
                p = Product(
                    product_name=f"SyncTest Batch Product {i} Waterproof Smart Watch",
                    description=f"Batch product {i} fitness smartwatch with heart rate monitoring, GPS tracking, and 14-day battery life.",
                    brand="SyncTestBrand",
                    category="Electronics",
                    tags=f"smartwatch, fitness, batch{i}, synctest",
                    price=Decimal(f"{99 + i}.00"),
                    image="https://example.com/watch.jpg",
                )
                db.add(p)
                db.commit()
                db.refresh(p)
                batch_ids.append(p.id)

            # Run sync with batch_size=2
            res = synchronize_embeddings(db, batch_size=2)
            assert res.new_count == 4, f"Expected 4 new embeddings, got {res.new_count}"
            assert res.updated_count == 0, f"Expected 0 updated, got {res.updated_count}"
            assert res.deleted_count == 0, f"Expected 0 deleted, got {res.deleted_count}"

            # Verify all 4 are in ChromaDB
            chroma_items = collection.get(ids=[str(bid) for bid in batch_ids])
            assert len(chroma_items["ids"]) == 4, f"Expected 4 vectors, got {len(chroma_items['ids'])}"

            # Cleanup
            for bid in batch_ids:
                p = db.get(Product, bid)
                if p:
                    db.delete(p)
            db.commit()

            # Clean sync
            res_clean = synchronize_embeddings(db, batch_size=2)
            assert res_clean.deleted_count == 4, f"Expected 4 deletions, got {res_clean.deleted_count}"
            print(f"    Batch processing (batch_size=2) verified across 4 additions and 4 deletions.")

        runner.run("6. Multi-Product Batch Embedding Generation", test_batch_processing)

        # -------------------------------------------------------------
        # Test 7: Concurrency Non-Blocking Guard
        # -------------------------------------------------------------
        def test_concurrency_guard():
            from app.services.embedding_sync import _sync_lock, is_sync_in_progress
            # Acquire lock manually
            assert not is_sync_in_progress()
            _sync_lock.acquire()
            try:
                assert is_sync_in_progress()
                # Run sync with non_blocking=True
                res = synchronize_embeddings(db, non_blocking=True)
                assert "Sync already in progress" in res.errors or len(res.errors) > 0
                assert res.has_changes is False
            finally:
                _sync_lock.release()
            print("    Concurrency lock guard correctly prevented simultaneous sync executions.")

        runner.run("7. Concurrency Lock & Non-Blocking Safety", test_concurrency_guard)

    finally:
        db.close()

    print("\n" + "=" * 80)
    print(f"TEST RUN COMPLETE: {runner.passed} PASSED, {runner.failed} FAILED (TOTAL: {runner.passed + runner.failed})")
    print("=" * 80)

    if runner.failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
