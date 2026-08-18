#!/bin/sh
set -e

echo "=================================================="
echo "Starting NorthStar Product Search Backend Service"
echo "=================================================="

# 1. Wait for PostgreSQL to be ready and accept connections
echo "[1/4] Waiting for database connection..."
python - <<'EOF'
import sys
import time
from sqlalchemy import create_engine
from app.core.config import settings

max_retries = 30
retry_interval = 2

engine = create_engine(settings.DATABASE_URL)
for i in range(1, max_retries + 1):
    try:
        with engine.connect() as conn:
            print(f"[OK] Database connection established successfully on attempt {i}.")
            sys.exit(0)
    except Exception as e:
        print(f"[WAIT] Database not ready yet (attempt {i}/{max_retries}): {e}")
        time.sleep(retry_interval)

print("[ERROR] Database connection failed after maximum retries.")
sys.exit(1)
EOF

# 2. Run Alembic database migrations (creates schema, pg_trgm extension, indexes)
echo "[2/4] Applying database migrations (alembic upgrade head)..."
alembic upgrade head

# 3. Seed / Import product data (idempotent duplicate prevention)
echo "[3/4] Checking and importing product dataset..."
python scripts/import_products.py

# 4. Start Uvicorn ASGI server
echo "[4/4] Starting FastAPI Uvicorn server on 0.0.0.0:8000..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
