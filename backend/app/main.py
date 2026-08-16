from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import search_router
from app.core.database import engine

app = FastAPI(
    title="NorthStar Product Search API",
    version="1.0.0",
)

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