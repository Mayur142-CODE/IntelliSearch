"""add pg_trgm and trigram indexes

Revision ID: 002_pg_trgm_fuzzy_search
Revises: 001_initial_products
Create Date: 2026-08-16 12:28:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_pg_trgm_fuzzy_search'
down_revision: Union[str, None] = '001_initial_products'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable PostgreSQL pg_trgm extension
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # 2. Create GIN trigram indexes for fast fuzzy text searching
    op.execute(
        "CREATE INDEX IF NOT EXISTS gin_products_product_name_trgm "
        "ON products USING gin (product_name gin_trgm_ops);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS gin_products_brand_trgm "
        "ON products USING gin (brand gin_trgm_ops);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS gin_products_category_trgm "
        "ON products USING gin (category gin_trgm_ops);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS gin_products_tags_trgm "
        "ON products USING gin (tags gin_trgm_ops);"
    )


def downgrade() -> None:
    # Drop trigram GIN indexes
    op.execute("DROP INDEX IF EXISTS gin_products_tags_trgm;")
    op.execute("DROP INDEX IF EXISTS gin_products_category_trgm;")
    op.execute("DROP INDEX IF EXISTS gin_products_brand_trgm;")
    op.execute("DROP INDEX IF EXISTS gin_products_product_name_trgm;")

    # Drop pg_trgm extension
    op.execute("DROP EXTENSION IF EXISTS pg_trgm;")
