import asyncio
import logging
import time
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.api import search_router, autocomplete_router
from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.services.embedding_sync import synchronize_embeddings, is_sync_in_progress
from app.services.query_parser import CatalogVocabulary

logger = logging.getLogger(__name__)

# Global handle for the background periodic synchronization task
_sync_task: Optional[asyncio.Task] = None


async def _periodic_embedding_sync_worker():
    """Non-blocking background loop that periodically synchronizes product embeddings."""
    interval = max(10, settings.EMBEDDING_SYNC_INTERVAL_SECONDS)
    logger.info(f"[embedding_sync_worker] Periodic sync worker started (interval: {interval}s).")

    while True:
        try:
            await asyncio.sleep(interval)
            logger.debug("[embedding_sync_worker] Triggering scheduled incremental embedding sync...")
            
            # Run sync in threadpool to avoid blocking FastAPI event loop / search requests
            loop = asyncio.get_running_loop()
            db = SessionLocal()
            try:
                result = await loop.run_in_executor(
                    None,
                    synchronize_embeddings,
                    db,
                    settings.EMBEDDING_BATCH_SIZE,
                    False,  # force_rebuild
                    True,   # non_blocking
                )
                if result.has_changes:
                    # Also reload catalog vocabulary if products were added/modified
                    vocab = CatalogVocabulary.get_instance()
                    vocab.load(db)
                    logger.info(f"[embedding_sync_worker] Catalog vocabulary refreshed after sync changes.")
            finally:
                db.close()

        except asyncio.CancelledError:
            logger.info("[embedding_sync_worker] Background sync task received cancellation. Exiting worker.")
            break
        except Exception as e:
            logger.error(f"[embedding_sync_worker] Error during periodic sync: {e}", exc_info=True)


app = FastAPI(
    title="NorthStar Product Search API",
    version="1.2.0",
)


@app.on_event("startup")
async def startup_warmup():
    """
    Warmup application resources at startup:
    1. Load dynamic catalog vocabulary from PostgreSQL (brands, categories).
    2. Synchronize ChromaDB vector collection incrementally.
    3. Pre-load FastEmbed ONNX model & prime inference JIT.
    4. Start periodic background synchronization worker.
    """
    global _sync_task
    t0 = time.perf_counter()
    try:
        # 1. Warmup Catalog Vocabulary
        db = SessionLocal()
        try:
            vocab = CatalogVocabulary.get_instance()
            vocab.load(db)
            logger.info(f"[startup] Catalog vocabulary loaded: {len(vocab.brands)} brands, {len(vocab.categories)} categories.")
            
            # 2. Incremental Sync at Startup
            sync_res = synchronize_embeddings(db, batch_size=settings.EMBEDDING_BATCH_SIZE, non_blocking=False)
            logger.info(f"[startup] Embedding sync completed: +{sync_res.new_count} new, ~{sync_res.updated_count} updated, -{sync_res.deleted_count} deleted.")
        finally:
            db.close()

        # 3. Warmup ChromaDB + FastEmbed ONNX Model
        from app.services.semantic_search import get_semantic_search_resources, _get_query_embedding
        get_semantic_search_resources()
        _get_query_embedding("warmup query")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[startup] ChromaDB + model loaded and warmed up in {elapsed_ms:.1f} ms")

        # 4. Start Background Periodic Sync Task
        if settings.EMBEDDING_SYNC_ENABLED:
            _sync_task = asyncio.create_task(_periodic_embedding_sync_worker())
            logger.info(f"[startup] Scheduled periodic embedding sync every {settings.EMBEDDING_SYNC_INTERVAL_SECONDS}s.")
        else:
            logger.info("[startup] Periodic embedding sync is disabled via configuration.")

    except Exception as e:
        logger.warning(f"[startup] Could not complete full startup warmup: {e}")


@app.on_event("shutdown")
async def shutdown_cleanup():
    """Clean up background tasks on application shutdown."""
    global _sync_task
    if _sync_task and not _sync_task.done():
        logger.info("[shutdown] Cancelling periodic embedding sync worker...")
        _sync_task.cancel()
        try:
            await _sync_task
        except asyncio.CancelledError:
            pass


# Configure CORS for local React development server
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(autocomplete_router)


@app.get("/")
def root():
    return {
        "message": "NorthStar Product Search API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/db-test")
def database_test():
    try:
        with engine.connect():
            return {
                "status": "success",
                "message": "PostgreSQL connection successful",
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/api/sync-embeddings")
def trigger_embedding_sync(background_tasks: BackgroundTasks):
    """Trigger manual background synchronization of product embeddings."""
    if is_sync_in_progress():
        return {
            "status": "in_progress",
            "message": "Embedding synchronization is already running.",
        }

    def _run():
        db = SessionLocal()
        try:
            synchronize_embeddings(db, batch_size=settings.EMBEDDING_BATCH_SIZE)
        finally:
            db.close()

    background_tasks.add_task(_run)
    return {
        "status": "triggered",
        "message": "Incremental embedding synchronization started in background.",
    }


@app.get("/api/sync-status")
def get_sync_status():
    """Get the current embedding synchronization state."""
    return {
        "is_sync_running": is_sync_in_progress(),
        "sync_enabled": settings.EMBEDDING_SYNC_ENABLED,
        "sync_interval_seconds": settings.EMBEDDING_SYNC_INTERVAL_SECONDS,
        "batch_size": settings.EMBEDDING_BATCH_SIZE,
    }