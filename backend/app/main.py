import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import search_router
from app.core.database import engine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NorthStar Product Search API",
    version="1.0.0",
)


@app.on_event("startup")
def startup_warmup():
    """
    Pre-load FastEmbed model and embedding matrix at application startup.

    Why: Without pre-loading, the first search request pays the full cold-start cost
    (~277ms model load + ~20ms first inference). By loading eagerly at startup,
    all search requests receive warm latency from the very first request.

    Also runs one warm-up inference pass to prime the ONNX Runtime JIT compiler,
    ensuring subsequent inferences hit steady-state performance (~17-22ms per query).
    """
    t0 = time.perf_counter()
    try:
        from app.services.semantic_search import get_semantic_search_resources, _get_query_embedding
        get_semantic_search_resources()
        # One warm-up inference to prime ONNX runtime JIT
        _get_query_embedding("warmup")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[startup] Model + embeddings loaded and warmed up in {elapsed_ms:.1f} ms")
    except Exception as e:
        logger.warning(f"[startup] Could not pre-load semantic resources: {e}")


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