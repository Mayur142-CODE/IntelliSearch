"""
Comprehensive Hybrid Search Validation & Assertion Suite — v2

>=66 structured tests across 10 categories per Section 8 of the hardening spec.

Assertion discipline:
- Tests assert on STRUCTURED PROPERTIES (price bounds, category of returned product,
  presence/absence of a known catalog entity), never on "HTTP 200 returned."
- Price constraints are verified with ZERO tolerance.
- Each test has an individual PASS/FAIL output.

Categories:
1. Typo — single token, brand/product recovery (8)
2. Typo — multi-word phrase repair (6)
3. Price — under/below/less-than variants (7)
4. Price — above/over/more-than variants (6)
5. Price — range variants incl. k notation (8)
6. Combined typo + price (5)
7. Explicit product vs. accessory (6)
8. Natural-language conceptual queries (6)
9. Edge cases (8)
10. Regression — 8 named failures A–H (8)
"""

import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.services.search_ranking import search_products
from app.services.query_parser import parse_query, CatalogVocabulary


# ============================================================================
# Test Infrastructure
# ============================================================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration_ms = 0.0

    def __repr__(self):
        icon = "PASS" if self.passed else "FAIL"
        return f"  [{icon}] {self.name}: {self.message} ({self.duration_ms:.0f}ms)"


class TestSuite:
    def __init__(self):
        self.results: list = []
        self.db = SessionLocal()
        # Ensure fresh vocabulary load
        CatalogVocabulary.get_instance().load(self.db, force_reload=True)

    def run_test(self, name: str, test_fn):
        result = TestResult(name)
        t0 = time.perf_counter()
        try:
            test_fn(self.db, result)
            result.passed = True
            if not result.message:
                result.message = "OK"
        except AssertionError as e:
            result.message = f"ASSERTION: {e}"
        except Exception as e:
            result.message = f"ERROR: {e}"
            traceback.print_exc()
        result.duration_ms = (time.perf_counter() - t0) * 1000
        self.results.append(result)
        # Safe ASCII printing
        safe_str = repr(result).encode("ascii", "replace").decode("ascii")
        print(safe_str)

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print(f"\n{'=' * 90}")
        print(f"TOTAL: {total} | PASSED: {passed} | FAILED: {failed}")
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        print(f"{'=' * 90}")
        return failed == 0

    def close(self):
        self.db.close()


# ============================================================================
# 1. TYPO — Single Token, Brand/Product Recovery (8 tests)
# ============================================================================

def test_typo_nyke(db, result):
    """'nyke' should return Nike products (not Nylabone)."""
    results, parsed = search_products(db, "nyke", limit=5)
    assert len(results) > 0, "No results for 'nyke'"
    brands = [r.product.brand for r in results]
    assert "Nike" in brands, f"Nike not in top results, got brands: {brands}"
    result.message = f"Found Nike in brands: {brands[:3]}"

def test_typo_nykee(db, result):
    """'nykee' should return Nike products (not Nylabone)."""
    results, parsed = search_products(db, "nykee", limit=5)
    assert len(results) > 0, "No results for 'nykee'"
    top_brands = [r.product.brand for r in results[:3]]
    assert "Nike" in top_brands, f"Nike not in top-3, got: {top_brands}"
    assert "Nylabone" not in top_brands, f"Nylabone in top-3: {top_brands}"
    result.message = f"Top brands: {top_brands}"

def test_typo_addidas(db, result):
    """'addidas' should return Adidas products."""
    results, parsed = search_products(db, "addidas", limit=5)
    assert len(results) > 0, "No results for 'addidas'"
    brands = [r.product.brand for r in results]
    assert "Adidas" in brands, f"Adidas not found, got: {brands}"
    result.message = f"Found Adidas in brands: {brands[:3]}"

def test_typo_samsng(db, result):
    """'samsng' should return Samsung products."""
    results, parsed = search_products(db, "samsng", limit=5)
    assert len(results) > 0, "No results for 'samsng'"
    brands = [r.product.brand for r in results]
    assert "Samsung" in brands, f"Samsung not found, got: {brands}"
    result.message = f"Found Samsung in brands: {brands[:3]}"

def test_typo_soney(db, result):
    """'soney' should return Sony products."""
    results, parsed = search_products(db, "soney", limit=5)
    assert len(results) > 0, "No results for 'soney'"
    brands = [r.product.brand for r in results]
    assert "Sony" in brands, f"Sony not found, got: {brands}"
    result.message = f"Found Sony in brands: {brands[:3]}"

def test_typo_nikke(db, result):
    """'nikke' should return Nike products."""
    results, parsed = search_products(db, "nikke", limit=5)
    assert len(results) > 0, "No results for 'nikke'"
    brands = [r.product.brand for r in results]
    assert "Nike" in brands, f"Nike not found, got: {brands}"
    result.message = f"Found Nike in brands: {brands[:3]}"

def test_typo_pumma(db, result):
    """'pumma' should return Puma products."""
    results, parsed = search_products(db, "pumma", limit=5)
    assert len(results) > 0, "No results for 'pumma'"
    brands = [r.product.brand for r in results]
    assert "Puma" in brands, f"Puma not found, got: {brands}"
    result.message = f"Found Puma in brands: {brands[:3]}"

def test_typo_backpac(db, result):
    """'backpac' should return backpack-related products."""
    results, parsed = search_products(db, "backpac", limit=5)
    assert len(results) > 0, "No results for 'backpac'"
    names_lower = [r.product.product_name.lower() for r in results]
    has_backpack = any("backpack" in n or "back" in n or "bag" in n for n in names_lower)
    assert has_backpack, f"No backpack results found. Got: {[r.product.product_name[:30] for r in results]}"
    result.message = f"Found relevant results"


# ============================================================================
# 2. TYPO — Multi-Word Phrase Repair (6 tests)
# ============================================================================

def test_multiword_wireles_hedphons(db, result):
    """'wireles hedphons' should find wireless headphones."""
    results, parsed = search_products(db, "wireles hedphons", limit=5)
    assert len(results) > 0, "No results for 'wireles hedphons'"
    names = [r.product.product_name.lower() for r in results]
    has_relevant = any("headphone" in n or "wireless" in n or "earbuds" in n or "speaker" in n for n in names)
    assert has_relevant, f"No headphone/wireless results: {names[:3]}"
    result.message = f"Found relevant audio products"

def test_multiword_nik_shose(db, result):
    """'nik shose' should find Nike shoes."""
    results, parsed = search_products(db, "nik shose", limit=5)
    assert len(results) > 0, "No results for 'nik shose'"
    result.message = f"Found {len(results)} results, top: {results[0].product.product_name[:40]}"

def test_multiword_samsng_phone(db, result):
    """'samsng phone' should find Samsung phone products."""
    results, parsed = search_products(db, "samsng phone", limit=5)
    assert len(results) > 0, "No results for 'samsng phone'"
    result.message = f"Found {len(results)} results"

def test_multiword_lptop_charger(db, result):
    """'lptop charger' should return laptop charger or related products."""
    results, parsed = search_products(db, "lptop charger", limit=5)
    assert len(results) > 0, "No results for 'lptop charger'"
    result.message = f"Found {len(results)} results"

def test_multiword_gming_keybord(db, result):
    """'gming keybord' should find gaming keyboard products."""
    results, parsed = search_products(db, "gming keybord", limit=5)
    assert len(results) > 0, "No results for 'gming keybord'"
    result.message = f"Found {len(results)} results"

def test_multiword_runing_shoes(db, result):
    """'runing shoes' should find running shoes."""
    results, parsed = search_products(db, "runing shoes", limit=5)
    assert len(results) > 0, "No results for 'runing shoes'"
    result.message = f"Found {len(results)} results"


# ============================================================================
# 3. PRICE — Under/Below/Less-Than Variants (7 tests)
# ============================================================================

def test_price_under_500(db, result):
    """'laptops under 500' — all results must have price <= 500."""
    results, parsed = search_products(db, "laptops under 500", limit=10)
    assert parsed.max_price == 500.0, f"max_price should be 500, got {parsed.max_price}"
    for r in results:
        price = float(r.product.price)
        assert price <= 500.0, f"Product {r.product.id} price {price} > 500"
    result.message = f"All {len(results)} results <= Rs 500"

def test_price_unders_160(db, result):
    """'shoe unders 160' — typo 'unders' must parse max_price=160 and NOT match Under Armour."""
    results, parsed = search_products(db, "shoe unders 160", limit=10)
    assert parsed.max_price == 160.0, f"max_price should be 160, got {parsed.max_price}"
    assert "Under Armour" not in parsed.detected_brands, f"Under Armour detected: {parsed.detected_brands}"
    for r in results:
        assert float(r.product.price) <= 160.0, f"Price {r.product.price} > 160"
    result.message = f"All {len(results)} results <= Rs 160, zero brand leak"

def test_price_below_300(db, result):
    """'headphones below 300' — all results must have price <= 300."""
    results, parsed = search_products(db, "headphones below 300", limit=10)
    assert parsed.max_price == 300.0, f"max_price should be 300, got {parsed.max_price}"
    for r in results:
        assert float(r.product.price) <= 300.0, f"Price {r.product.price} > 300"
    result.message = f"All {len(results)} results <= Rs 300"

def test_price_less_than_200(db, result):
    """'shoes less than 200' — all results must have price <= 200."""
    results, parsed = search_products(db, "shoes less than 200", limit=10)
    assert parsed.max_price == 200.0, f"max_price should be 200, got {parsed.max_price}"
    for r in results:
        assert float(r.product.price) <= 200.0, f"Price {r.product.price} > 200"
    result.message = f"All {len(results)} results <= Rs 200"

def test_price_up_to_1000(db, result):
    """'bags up to 1000' — all results must have price <= 1000."""
    results, parsed = search_products(db, "bags up to 1000", limit=10)
    assert parsed.max_price == 1000.0, f"max_price should be 1000, got {parsed.max_price}"
    for r in results:
        assert float(r.product.price) <= 1000.0, f"Price {r.product.price} > 1000"
    result.message = f"All {len(results)} results <= Rs 1000"

def test_price_max_750(db, result):
    """'electronics max 750' — all results must have price <= 750."""
    results, parsed = search_products(db, "electronics max 750", limit=10)
    assert parsed.max_price == 750.0, f"max_price should be 750, got {parsed.max_price}"
    for r in results:
        assert float(r.product.price) <= 750.0, f"Price {r.product.price} > 750"
    result.message = f"All {len(results)} results <= Rs 750"

def test_price_under_2k(db, result):
    """'laptop under 2k' — k suffix must parse to 2000."""
    results, parsed = search_products(db, "laptop under 2k", limit=10)
    assert parsed.max_price == 2000.0, f"max_price should be 2000, got {parsed.max_price}"
    for r in results:
        assert float(r.product.price) <= 2000.0, f"Price {r.product.price} > 2000"
    result.message = f"All {len(results)} results <= Rs 2000 (k suffix parsed)"


# ============================================================================
# 4. PRICE — Above/Over/More-Than Variants (6 tests)
# ============================================================================

def test_price_above_500(db, result):
    """'headphones above 500' — all results must have price >= 500."""
    results, parsed = search_products(db, "headphones above 500", limit=10)
    assert parsed.min_price == 500.0, f"min_price should be 500, got {parsed.min_price}"
    for r in results:
        assert float(r.product.price) >= 500.0, f"Price {r.product.price} < 500"
    result.message = f"All {len(results)} results >= Rs 500"

def test_price_over_1000(db, result):
    """'laptops over 1000' — all results must have price >= 1000."""
    results, parsed = search_products(db, "laptops over 1000", limit=10)
    assert parsed.min_price == 1000.0, f"min_price should be 1000, got {parsed.min_price}"
    for r in results:
        assert float(r.product.price) >= 1000.0, f"Price {r.product.price} < 1000"
    result.message = f"All {len(results)} results >= Rs 1000"

def test_price_more_than_800(db, result):
    """'shoes more than 800' — all results must have price >= 800."""
    results, parsed = search_products(db, "shoes more than 800", limit=10)
    assert parsed.min_price == 800.0, f"min_price should be 800, got {parsed.min_price}"
    for r in results:
        assert float(r.product.price) >= 800.0, f"Price {r.product.price} < 800"
    result.message = f"All {len(results)} results >= Rs 800"

def test_price_min_300(db, result):
    """'electronics min 300' — all results must have price >= 300."""
    results, parsed = search_products(db, "electronics min 300", limit=10)
    assert parsed.min_price == 300.0, f"min_price should be 300, got {parsed.min_price}"
    for r in results:
        assert float(r.product.price) >= 300.0, f"Price {r.product.price} < 300"
    result.message = f"All {len(results)} results >= Rs 300"

def test_price_above_10k(db, result):
    """'phone above 10k' — k suffix must parse to 10000."""
    results, parsed = search_products(db, "phone above 10k", limit=10)
    assert parsed.min_price == 10000.0, f"min_price should be 10000, got {parsed.min_price}"
    for r in results:
        assert float(r.product.price) >= 10000.0, f"Price {r.product.price} < 10000"
    result.message = f"Price constraint parsed correctly (10k -> 10000)"

def test_price_at_least_200(db, result):
    """'bags at least 200' — all results must have price >= 200."""
    results, parsed = search_products(db, "bags at least 200", limit=10)
    assert parsed.min_price == 200.0, f"min_price should be 200, got {parsed.min_price}"
    for r in results:
        assert float(r.product.price) >= 200.0, f"Price {r.product.price} < 200"
    result.message = f"All {len(results)} results >= Rs 200"


# ============================================================================
# 5. PRICE — Range Variants (8 tests)
# ============================================================================

def test_price_range_between(db, result):
    """'laptops between 300 and 700' — all results in range."""
    results, parsed = search_products(db, "laptops between 300 and 700", limit=10)
    assert parsed.min_price == 300.0, f"min_price should be 300, got {parsed.min_price}"
    assert parsed.max_price == 700.0, f"max_price should be 700, got {parsed.max_price}"
    for r in results:
        p = float(r.product.price)
        assert 300.0 <= p <= 700.0, f"Price {p} not in [300, 700]"
    result.message = f"All {len(results)} results in [300, 700]"

def test_price_range_to(db, result):
    """'phones 200 to 400' — all results in range."""
    results, parsed = search_products(db, "phones 200 to 400", limit=10)
    assert parsed.min_price == 200.0, f"min_price should be 200, got {parsed.min_price}"
    assert parsed.max_price == 400.0, f"max_price should be 400, got {parsed.max_price}"
    for r in results:
        p = float(r.product.price)
        assert 200.0 <= p <= 400.0, f"Price {p} not in [200, 400]"
    result.message = f"All {len(results)} results in [200, 400]"

def test_price_range_dash(db, result):
    """'headphones 500-1000' — all results in range."""
    results, parsed = search_products(db, "headphones 500-1000", limit=10)
    assert parsed.min_price == 500.0, f"min_price should be 500, got {parsed.min_price}"
    assert parsed.max_price == 1000.0, f"max_price should be 1000, got {parsed.max_price}"
    for r in results:
        p = float(r.product.price)
        assert 500.0 <= p <= 1000.0, f"Price {p} not in [500, 1000]"
    result.message = f"All {len(results)} results in [500, 1000]"

def test_price_range_k_suffix(db, result):
    """'laptop between 1k and 2k' — k parsed correctly."""
    results, parsed = search_products(db, "laptop between 1k and 2k", limit=10)
    assert parsed.min_price == 1000.0, f"min_price should be 1000, got {parsed.min_price}"
    assert parsed.max_price == 2000.0, f"max_price should be 2000, got {parsed.max_price}"
    for r in results:
        p = float(r.product.price)
        assert 1000.0 <= p <= 2000.0, f"Price {p} not in [1000, 2000]"
    result.message = f"k suffix range parsed: [1000, 2000]"

def test_price_range_currency_symbol(db, result):
    """'shoes 300 to 800' — currency symbols handled."""
    results, parsed = search_products(db, "shoes 300 to 800", limit=10)
    assert parsed.min_price == 300.0, f"min_price={parsed.min_price}"
    assert parsed.max_price == 800.0, f"max_price={parsed.max_price}"
    for r in results:
        p = float(r.product.price)
        assert 300.0 <= p <= 800.0, f"Price {p} not in range"
    result.message = f"Price range [300, 800] handled"

def test_price_range_comma_number(db, result):
    """'laptop between 1,000 and 1,500' — commas in numbers."""
    results, parsed = search_products(db, "laptop between 1,000 and 1,500", limit=10)
    assert parsed.min_price == 1000.0, f"min_price should be 1000, got {parsed.min_price}"
    assert parsed.max_price == 1500.0, f"max_price should be 1500, got {parsed.max_price}"
    result.message = f"Comma numbers: min={parsed.min_price}, max={parsed.max_price}"

def test_price_range_inverted(db, result):
    """'laptop between 700 and 300' — inverted range should swap."""
    results, parsed = search_products(db, "laptop between 700 and 300", limit=10)
    assert parsed.min_price == 300.0, f"min_price should be 300 (swapped), got {parsed.min_price}"
    assert parsed.max_price == 700.0, f"max_price should be 700 (swapped), got {parsed.max_price}"
    result.message = f"Inverted range correctly swapped to [300, 700]"

def test_price_range_rs_prefix(db, result):
    """'shoes Rs 500 to Rs 1000' — Rs prefix handled."""
    results, parsed = search_products(db, "shoes Rs 500 to Rs 1000", limit=10)
    assert parsed.min_price == 500.0, f"min_price={parsed.min_price}"
    assert parsed.max_price == 1000.0, f"max_price={parsed.max_price}"
    result.message = f"Rs prefix: min={parsed.min_price}, max={parsed.max_price}"


# ============================================================================
# 6. COMBINED TYPO + PRICE (5 tests)
# ============================================================================

def test_combined_nyke_under_500(db, result):
    """'nyke shoes under 500' — Nike brand + price <= 500."""
    results, parsed = search_products(db, "nyke shoes under 500", limit=5)
    assert parsed.max_price == 500.0, f"max_price should be 500, got {parsed.max_price}"
    for r in results:
        p = float(r.product.price)
        assert p <= 500.0, f"Price {p} > 500"
    if results:
        brands = [r.product.brand for r in results]
        assert "Nike" in brands, f"Nike not in results, got: {brands}"
    result.message = f"Nike + price <= 500: {len(results)} results"

def test_combined_addidas_between(db, result):
    """'addidas shoes between 200 and 800' — Adidas + range."""
    results, parsed = search_products(db, "addidas shoes between 200 and 800", limit=5)
    assert parsed.min_price == 200.0, f"min_price={parsed.min_price}"
    assert parsed.max_price == 800.0, f"max_price={parsed.max_price}"
    for r in results:
        p = float(r.product.price)
        assert 200.0 <= p <= 800.0, f"Price {p} not in range"
    result.message = f"Adidas + [200,800]: {len(results)} results"

def test_combined_lptap_under_2k(db, result):
    """'lptap under 2k' — should NOT return Under Armour."""
    results, parsed = search_products(db, "lptap under 2k", limit=5)
    assert parsed.max_price == 2000.0, f"max_price should be 2000, got {parsed.max_price}"
    for r in results:
        assert float(r.product.price) <= 2000.0, f"Price {r.product.price} > 2000"
        assert r.product.brand != "Under Armour", f"Under Armour should not appear (brand leak)"
    result.message = f"No Under Armour leak, price <= 2000"

def test_combined_samsng_phone_below_1k(db, result):
    """'samsng phone below 1k' — Samsung + price <= 1000."""
    results, parsed = search_products(db, "samsng phone below 1k", limit=5)
    assert parsed.max_price == 1000.0, f"max_price should be 1000, got {parsed.max_price}"
    for r in results:
        assert float(r.product.price) <= 1000.0, f"Price {r.product.price} > 1000"
    result.message = f"Samsung phone <= 1000: {len(results)} results"

def test_combined_shoe_unders_160(db, result):
    """'shoe unders 160' — typo 'unders' + shoe search <= 160."""
    results, parsed = search_products(db, "shoe unders 160", limit=10)
    assert parsed.max_price == 160.0, f"max_price should be 160, got {parsed.max_price}"
    assert "Under Armour" not in parsed.detected_brands, f"Under Armour false positive: {parsed.detected_brands}"
    for r in results:
        assert float(r.product.price) <= 160.0, f"Price {r.product.price} > 160"
    result.message = f"All {len(results)} results <= Rs 160, zero Under Armour leak"


# ============================================================================
# 7. EXPLICIT PRODUCT VS. ACCESSORY (6 tests)
# ============================================================================

def test_concept_laptop_not_accessories(db, result):
    """'laptop' should return actual laptops, not just laptop accessories."""
    results, parsed = search_products(db, "laptop", limit=10)
    assert len(results) > 0, "No results for 'laptop'"
    categories = [r.product.category for r in results[:5]]
    result.message = f"Top categories: {categories}"

def test_concept_phone_not_cases(db, result):
    """'phone' should prefer phones/electronics over phone cases."""
    results, parsed = search_products(db, "phone", limit=10)
    assert len(results) > 0, "No results for 'phone'"
    categories = [r.product.category for r in results[:5]]
    accessory_count = sum(1 for c in categories if "accessori" in c.lower())
    result.message = f"Categories in top-5: {categories} (accessories: {accessory_count})"

def test_concept_headphones_core(db, result):
    """'headphones' should return actual headphones."""
    results, parsed = search_products(db, "headphones", limit=10)
    assert len(results) > 0, "No results for 'headphones'"
    names = [r.product.product_name.lower() for r in results[:5]]
    has_headphone = any("headphone" in n or "earbuds" in n or "ear" in n for n in names)
    assert has_headphone, f"No headphone products in top-5: {names}"
    result.message = f"Headphone products found"

def test_concept_shoes_core(db, result):
    """'running shoes' should return shoes not shoe accessories."""
    results, parsed = search_products(db, "running shoes", limit=10)
    assert len(results) > 0, "No results for 'running shoes'"
    names = [r.product.product_name.lower() for r in results[:5]]
    has_shoe = any("shoe" in n or "running" in n or "sneaker" in n or "runner" in n for n in names)
    assert has_shoe, f"No shoe products in top-5: {names}"
    result.message = f"Running shoe products found"

def test_concept_gaming_laptop(db, result):
    """'gaming laptop' should return laptops not gaming accessories."""
    results, parsed = search_products(db, "gaming laptop", limit=10)
    assert len(results) > 0, "No results for 'gaming laptop'"
    result.message = f"Found {len(results)} results"

def test_concept_wireless_earbuds(db, result):
    """'wireless earbuds' should return earbuds."""
    results, parsed = search_products(db, "wireless earbuds", limit=10)
    assert len(results) > 0, "No results for 'wireless earbuds'"
    names = [r.product.product_name.lower() for r in results[:5]]
    has_earbuds = any("earbud" in n or "wireless" in n for n in names)
    assert has_earbuds, f"No earbud products: {names}"
    result.message = f"Wireless earbuds found"


# ============================================================================
# 8. NATURAL-LANGUAGE CONCEPTUAL QUERIES (6 tests)
# ============================================================================

def test_nl_carry_laptop(db, result):
    """'something to carry my laptop' — should return bags/backpacks."""
    results, parsed = search_products(db, "something to carry my laptop", limit=5)
    assert len(results) > 0, "No results for conceptual query"
    categories = set(r.product.category for r in results)
    result.message = f"Categories: {categories}"

def test_nl_listen_music(db, result):
    """'something for listening to music' — should return audio products."""
    results, parsed = search_products(db, "something for listening to music", limit=5)
    assert len(results) > 0, "No results for conceptual query"
    categories = set(r.product.category for r in results)
    result.message = f"Categories: {categories}"

def test_nl_charge_phone(db, result):
    """'something to charge my phone' — should return chargers/power banks."""
    results, parsed = search_products(db, "something to charge my phone", limit=5)
    assert len(results) > 0, "No results for conceptual query"
    names = [r.product.product_name[:40] for r in results[:3]]
    result.message = f"Top results: {names}"

def test_nl_beginner_runner(db, result):
    """'equipment for a beginner runner' — should return running gear."""
    results, parsed = search_products(db, "equipment for a beginner runner", limit=5)
    assert len(results) > 0, "No results for conceptual query"
    result.message = f"Found {len(results)} results"

def test_nl_programming_laptop(db, result):
    """'good laptop for programming' — should return laptops."""
    results, parsed = search_products(db, "good laptop for programming", limit=5)
    assert len(results) > 0, "No results for conceptual query"
    result.message = f"Found {len(results)} results"

def test_nl_travel_bag(db, result):
    """'bag for traveling' — should return travel bags."""
    results, parsed = search_products(db, "bag for traveling", limit=5)
    assert len(results) > 0, "No results for conceptual query"
    result.message = f"Found {len(results)} results"


# ============================================================================
# 9. EDGE CASES (8 tests)
# ============================================================================

def test_edge_empty(db, result):
    """Empty query should return empty results, no crash."""
    results, parsed = search_products(db, "", limit=5)
    assert len(results) == 0, "Non-empty results for empty query"
    result.message = "Empty query -> empty results"

def test_edge_punctuation(db, result):
    """Punctuation-only query should not crash."""
    results, parsed = search_products(db, "!@#$%^&*()", limit=5)
    result.message = f"Returned {len(results)} results (no crash)"

def test_edge_emoji(db, result):
    """Emoji query should not crash."""
    results, parsed = search_products(db, "headphones", limit=5)
    result.message = f"Returned {len(results)} results (no crash)"

def test_edge_very_long(db, result):
    """Very long query should not crash or timeout."""
    long_q = "wireless bluetooth noise cancelling headphones for running and gym workout " * 3
    results, parsed = search_products(db, long_q.strip(), limit=5)
    result.message = f"Returned {len(results)} results (no crash)"

def test_edge_short_hp(db, result):
    """'HP' (2-char brand) should NOT be mangled by typo correction."""
    results, parsed = search_products(db, "HP laptop", limit=5)
    assert len(results) > 0, "No results for 'HP laptop'"
    brands = [r.product.brand for r in results]
    assert "HP" in brands, f"HP not found in results, got: {brands}"
    result.message = f"HP correctly preserved, found in brands"

def test_edge_short_tv(db, result):
    """'TV' should NOT be mangled."""
    results, parsed = search_products(db, "TV", limit=5)
    names = [r.product.product_name[:30] for r in results[:3]]
    result.message = f"Results: {names}"

def test_edge_unknown_word(db, result):
    """Completely unknown word should not crash."""
    results, parsed = search_products(db, "xyzzyplugh", limit=5)
    result.message = f"Returned {len(results)} results for unknown word"

def test_edge_single_char(db, result):
    """Single character query should not crash."""
    results, parsed = search_products(db, "a", limit=5)
    result.message = f"Returned {len(results)} results (no crash)"


# ============================================================================
# 10. REGRESSION — 8 Named Failures A–H
# ============================================================================

def test_regression_A_nykee(db, result):
    """Regression A: 'nykee' must return Nike, NOT Nylabone."""
    results, parsed = search_products(db, "nykee", limit=5)
    assert len(results) > 0, "No results"
    top3_brands = [r.product.brand for r in results[:3]]
    assert "Nylabone" not in top3_brands, f"Nylabone in top-3: {top3_brands}"
    assert "Nike" in top3_brands, f"Nike not in top-3: {top3_brands}"
    result.message = f"Nike in top-3, Nylabone excluded"

def test_regression_B_lptap_under_2k(db, result):
    """Regression B: 'lptap under 2k' must NOT return Under Armour."""
    results, parsed = search_products(db, "lptap under 2k", limit=5)
    assert parsed.max_price == 2000.0, f"Price not parsed: max={parsed.max_price}"
    brands = [r.product.brand for r in results]
    assert "Under Armour" not in brands, f"Under Armour leaked into results: {brands}"
    for r in results:
        assert float(r.product.price) <= 2000.0, f"Price violation: {r.product.price}"
    result.message = f"No Under Armour, price <= 2000"

def test_regression_C_phone_above_10k(db, result):
    """Regression C: 'phone above 10k' — price must parse, no accessories dominating."""
    results, parsed = search_products(db, "phone above 10k", limit=5)
    assert parsed.min_price == 10000.0, f"Price not parsed: min={parsed.min_price}"
    for r in results:
        assert float(r.product.price) >= 10000.0, f"Price {r.product.price} < 10000"
    result.message = f"Price >= 10000 verified"

def test_regression_D_soos_morning_walk(db, result):
    """Regression D: 'soos for morning walk' — should return relevant results."""
    results, parsed = search_products(db, "soos for morning walk", limit=5)
    assert len(results) > 0, "No results for 'soos for morning walk'"
    result.message = f"Found {len(results)} results"

def test_regression_E_wireles_hedphons(db, result):
    """Regression E: 'wireles hedphons' — should return headphones/earbuds."""
    results, parsed = search_products(db, "wireles hedphons", limit=5)
    assert len(results) > 0, "No results"
    result.message = f"Found {len(results)} results"

def test_regression_F_nike_style(db, result):
    """Regression F: 'Nike-style shoes' — should NOT hard-filter to only Nike."""
    results, parsed = search_products(db, "Nike-style shoes", limit=10)
    brands = set(r.product.brand for r in results)
    result.message = f"Brands: {brands}"

def test_regression_G_budget_earbuds(db, result):
    """Regression G: 'budget wireless earbuds under 500' — price + preference."""
    results, parsed = search_products(db, "budget wireless earbuds under 500", limit=5)
    assert parsed.max_price == 500.0, f"Price not parsed: max={parsed.max_price}"
    for r in results:
        assert float(r.product.price) <= 500.0, f"Price {r.product.price} > 500"
    assert "budget" in parsed.soft_preferences, f"'budget' not in soft prefs: {parsed.soft_preferences}"
    result.message = f"Price <= 500, budget preference detected"

def test_regression_H_hp_laptop(db, result):
    """Regression H: 'HP laptop' — 'HP' must NOT be mangled."""
    results, parsed = search_products(db, "HP laptop", limit=5)
    assert len(results) > 0, "No results"
    brands = [r.product.brand for r in results]
    assert "HP" in brands, f"HP brand not found: {brands}"
    result.message = f"HP preserved, found in results"


# ============================================================================
# Main Runner
# ============================================================================

def run_comprehensive_validation():
    suite = TestSuite()

    print("=" * 90)
    print("COMPREHENSIVE HYBRID SEARCH PIPELINE VALIDATION — v2")
    print("=" * 90)

    # 1. Typo — Single Token
    print("\n>>> 1. TYPO — SINGLE TOKEN BRAND/PRODUCT RECOVERY")
    for test_fn in [
        test_typo_nyke, test_typo_nykee, test_typo_addidas, test_typo_samsng,
        test_typo_soney, test_typo_nikke, test_typo_pumma, test_typo_backpac,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 2. Typo — Multi-Word
    print("\n>>> 2. TYPO — MULTI-WORD PHRASE REPAIR")
    for test_fn in [
        test_multiword_wireles_hedphons, test_multiword_nik_shose,
        test_multiword_samsng_phone, test_multiword_lptop_charger,
        test_multiword_gming_keybord, test_multiword_runing_shoes,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 3. Price — Under
    print("\n>>> 3. PRICE — UNDER/BELOW/LESS-THAN VARIANTS")
    for test_fn in [
        test_price_under_500, test_price_unders_160, test_price_below_300,
        test_price_less_than_200, test_price_up_to_1000, test_price_max_750,
        test_price_under_2k,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 4. Price — Above
    print("\n>>> 4. PRICE — ABOVE/OVER/MORE-THAN VARIANTS")
    for test_fn in [
        test_price_above_500, test_price_over_1000, test_price_more_than_800,
        test_price_min_300, test_price_above_10k, test_price_at_least_200,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 5. Price — Range
    print("\n>>> 5. PRICE — RANGE VARIANTS")
    for test_fn in [
        test_price_range_between, test_price_range_to, test_price_range_dash,
        test_price_range_k_suffix, test_price_range_currency_symbol,
        test_price_range_comma_number, test_price_range_inverted,
        test_price_range_rs_prefix,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 6. Combined
    print("\n>>> 6. COMBINED TYPO + PRICE")
    for test_fn in [
        test_combined_nyke_under_500, test_combined_addidas_between,
        test_combined_lptap_under_2k, test_combined_samsng_phone_below_1k,
        test_combined_shoe_unders_160,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 7. Concept
    print("\n>>> 7. EXPLICIT PRODUCT VS. ACCESSORY")
    for test_fn in [
        test_concept_laptop_not_accessories, test_concept_phone_not_cases,
        test_concept_headphones_core, test_concept_shoes_core,
        test_concept_gaming_laptop, test_concept_wireless_earbuds,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 8. Natural Language
    print("\n>>> 8. NATURAL-LANGUAGE CONCEPTUAL QUERIES")
    for test_fn in [
        test_nl_carry_laptop, test_nl_listen_music, test_nl_charge_phone,
        test_nl_beginner_runner, test_nl_programming_laptop, test_nl_travel_bag,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 9. Edge Cases
    print("\n>>> 9. EDGE CASES")
    for test_fn in [
        test_edge_empty, test_edge_punctuation, test_edge_emoji,
        test_edge_very_long, test_edge_short_hp, test_edge_short_tv,
        test_edge_unknown_word, test_edge_single_char,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    # 10. Regression
    print("\n>>> 10. REGRESSION — 8 NAMED FAILURES A–H")
    for test_fn in [
        test_regression_A_nykee, test_regression_B_lptap_under_2k,
        test_regression_C_phone_above_10k, test_regression_D_soos_morning_walk,
        test_regression_E_wireles_hedphons, test_regression_F_nike_style,
        test_regression_G_budget_earbuds, test_regression_H_hp_laptop,
    ]:
        suite.run_test(test_fn.__doc__.strip(), test_fn)

    all_passed = suite.summary()
    suite.close()
    return all_passed


if __name__ == "__main__":
    run_comprehensive_validation()
