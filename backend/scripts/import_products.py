import csv
from decimal import Decimal, InvalidOperation   
from pathlib import Path
import sys

# Ensure backend directory is in sys.path for app module imports
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select
from app.core.database import SessionLocal
from app.models.product import Product

REQUIRED_COLUMNS = [
    "Product Name",
    "Description",
    "Brand",
    "Category",
    "Tags",
    "Price",
    "Image",
]


def resolve_csv_path() -> Path:
    """Find products.csv in standard locations relative to backend directory."""
    candidates = [
        BACKEND_DIR / "data" / "products.csv",
        BACKEND_DIR / "app" / "data" / "products.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate products.csv in expected paths: {candidates}"
    )


def normalize_name(name: str) -> str:
    """Normalize product name for case-insensitive and whitespace-insensitive matching."""
    return " ".join(name.strip().lower().split())


def import_products():
    """Import product records from CSV into PostgreSQL database with duplicate prevention."""
    csv_path = resolve_csv_path()
    print(f"Reading dataset from: {csv_path}")

    session = SessionLocal()

    total_csv_rows = 0
    valid_rows = 0
    duplicate_csv_rows = 0
    existing_db_products = 0
    invalid_rows = 0
    newly_imported = 0

    invalid_row_details = []

    try:
        # Step 1: Pre-load existing product names from DB for duplicate detection
        print("Checking existing products in database...")
        existing_db_names = session.execute(
            select(Product.product_name)
        ).scalars().all()
        existing_db_keys = {
            normalize_name(name) for name in existing_db_names if name
        }
        print(f"Found {len(existing_db_keys)} existing unique product names in database.")

        seen_csv_keys = set()
        products_to_insert = []

        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            # Validate required columns
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or missing header row.")

            fieldnames = [field.strip() for field in reader.fieldnames if field]
            missing_headers = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
            if missing_headers:
                raise ValueError(
                    f"CSV missing required columns: {missing_headers}. Found: {reader.fieldnames}"
                )

            # Step 2: Process CSV rows
            for row_num, row in enumerate(reader, start=2):
                total_csv_rows += 1

                # Extract and clean raw values
                raw_name = (row.get("Product Name") or "").strip()
                raw_desc = (row.get("Description") or "").strip()
                raw_brand = (row.get("Brand") or "").strip()
                raw_cat = (row.get("Category") or "").strip()
                raw_tags = (row.get("Tags") or "").strip()
                raw_price = (row.get("Price") or "").strip()
                raw_img = (row.get("Image") or "").strip()

                # Check required non-empty text fields
                missing_fields = []
                if not raw_name:
                    missing_fields.append("Product Name")
                if not raw_desc:
                    missing_fields.append("Description")
                if not raw_brand:
                    missing_fields.append("Brand")
                if not raw_cat:
                    missing_fields.append("Category")
                if not raw_tags:
                    missing_fields.append("Tags")
                if not raw_price:
                    missing_fields.append("Price")
                if not raw_img:
                    missing_fields.append("Image")

                if missing_fields:
                    invalid_rows += 1
                    invalid_row_details.append(
                        f"Row {row_num}: Missing required field(s): {', '.join(missing_fields)}"
                    )
                    continue

                # Parse and validate Price numeric value
                try:
                    price_val = Decimal(raw_price)
                except (InvalidOperation, ValueError):
                    invalid_rows += 1
                    invalid_row_details.append(
                        f"Row {row_num}: Invalid numeric price value '{raw_price}'"
                    )
                    continue

                valid_rows += 1

                # Normalize name key for duplicate checking
                name_key = normalize_name(raw_name)

                # Check for in-CSV duplicates
                if name_key in seen_csv_keys:
                    duplicate_csv_rows += 1
                    continue
                seen_csv_keys.add(name_key)

                # Check for existing DB products
                if name_key in existing_db_keys:
                    existing_db_products += 1
                    continue

                # Prepare Product model instance
                product_obj = Product(
                    product_name=raw_name,
                    description=raw_desc,
                    brand=raw_brand,
                    category=raw_cat,
                    tags=raw_tags,
                    price=price_val,
                    image=raw_img,
                )
                products_to_insert.append(product_obj)

        # Step 3: Batch insertion inside a database transaction
        if products_to_insert:
            print(f"Inserting {len(products_to_insert)} new products into database...")
            batch_size = 500
            for i in range(0, len(products_to_insert), batch_size):
                batch = products_to_insert[i : i + batch_size]
                session.add_all(batch)
                session.flush()
            session.commit()
            newly_imported = len(products_to_insert)
        else:
            print("No new products to insert.")

        # Step 4: Display invalid row warnings if any exist
        if invalid_row_details:
            print("\n--- INVALID ROW WARNINGS ---")
            for detail in invalid_row_details:
                print(f"[WARNING] {detail}")

        # Step 5: Print summary report
        print("\n" + "=" * 40)
        print("PRODUCT IMPORT SUMMARY")
        print("=" * 40)
        print(f"CSV rows:              {total_csv_rows}")
        print(f"Valid rows:            {valid_rows}")
        print(f"Duplicate CSV rows:    {duplicate_csv_rows}")
        print(f"Existing DB products:  {existing_db_products}")
        print(f"Invalid rows:          {invalid_rows}")
        print(f"Newly imported:        {newly_imported}")
        print("\nImport completed successfully.")

    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] Import failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import_products()
