# ==============================================================================
# Offline Intelligent Product Search — Backend Dockerfile
# ==============================================================================

FROM python:3.12-slim

# Enforce Python output flushing and strict offline execution for model loaders
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Install system dependencies required for psycopg2 / PostgreSQL and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python package dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code, migration configs, scripts, models, and embeddings
COPY . .

# Ensure entrypoint script has executable permissions
RUN chmod +x entrypoint.sh

# Expose FastAPI application port
EXPOSE 8000

# Healthcheck to verify backend server readiness
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
