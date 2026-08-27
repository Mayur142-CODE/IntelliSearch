import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import search_router, autocomplete_router
from app.core.database import SessionLocal, engine
from app.services.query_parser import CatalogVocabulary

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NorthStar Product Search API",
    version="1.1.0",
)


@app.on_event("startup")
def startup_warmup():
    """
    Warmup application resources at startup:
    1. Load dynamic catalog vocabulary from PostgreSQL (brands, categories).
    2. Initialize ChromaDB vector collection.
    3. Pre-load FastEmbed ONNX model & prime inference JIT.
    """
    t0 = time.perf_counter()
    try:
        # 1. Warmup Catalog Vocabulary
        db = SessionLocal()
        try:
            vocab = CatalogVocabulary.get_instance()
            vocab.load(db)
            logger.info(f"[startup] Catalog vocabulary loaded: {len(vocab.brands)} brands, {len(vocab.categories)} categories.")
        finally:
            db.close()

        # 2. Warmup ChromaDB + FastEmbed ONNX Model
        from app.services.semantic_search import get_semantic_search_resources, _get_query_embedding
        get_semantic_search_resources()
        _get_query_embedding("warmup query")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[startup] ChromaDB + model loaded and warmed up in {elapsed_ms:.1f} ms")
    except Exception as e:
        logger.warning(f"[startup] Could not complete full startup warmup: {e}")


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