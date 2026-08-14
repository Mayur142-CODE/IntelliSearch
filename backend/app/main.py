from fastapi import FastAPI

app = FastAPI(
    title="NorthStar Product Search API",
    version="1.0.0",
)


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