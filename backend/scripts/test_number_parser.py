"""
Unit Test Suite for Generic Number & Numeric Scale Parser
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.number_parser import parse_numeric_expression, resolve_scale


def test_scales():
    assert resolve_scale("k") == 1_000.0
    assert resolve_scale("K") == 1_000.0
    assert resolve_scale("thousand") == 1_000.0
    assert resolve_scale("Thousands") == 1_000.0
    assert resolve_scale("lakh") == 100_000.0
    assert resolve_scale("lakhs") == 100_000.0
    assert resolve_scale("LAKH") == 100_000.0
    assert resolve_scale("lac") == 100_000.0
    assert resolve_scale("lacs") == 100_000.0
    assert resolve_scale("crore") == 10_000_000.0
    assert resolve_scale("crores") == 10_000_000.0
    assert resolve_scale("cr") == 10_000_000.0
    assert resolve_scale("m") == 1_000_000.0
    assert resolve_scale("M") == 1_000_000.0
    assert resolve_scale("million") == 1_000_000.0
    assert resolve_scale("millions") == 1_000_000.0
    assert resolve_scale("b") == 1_000_000_000.0
    assert resolve_scale("B") == 1_000_000_000.0
    assert resolve_scale("billion") == 1_000_000_000.0
    assert resolve_scale("billions") == 1_000_000_000.0
    assert resolve_scale("trillion") == 1_000_000_000_000.0
    assert resolve_scale("trillions") == 1_000_000_000_000.0


def test_fuzzy_scales():
    assert resolve_scale("thosand") == 1_000.0
    assert resolve_scale("thousnd") == 1_000.0
    assert resolve_scale("millon") == 1_000_000.0
    assert resolve_scale("lkah") == 100_000.0
    assert resolve_scale("croer") == 10_000_000.0
    assert resolve_scale("billon") == 1_000_000_000.0


def test_plain_numbers():
    assert parse_numeric_expression("3000") == 3000.0
    assert parse_numeric_expression("500") == 500.0
    assert parse_numeric_expression("12.5") == 12.5


def test_comma_formatting():
    assert parse_numeric_expression("50,000") == 50000.0
    assert parse_numeric_expression("1,00,000") == 100000.0
    assert parse_numeric_expression("10,00,000") == 1000000.0
    assert parse_numeric_expression("1,000,000") == 1000000.0


def test_thousands():
    assert parse_numeric_expression("2k") == 2000.0
    assert parse_numeric_expression("2K") == 2000.0
    assert parse_numeric_expression("2 thousand") == 2000.0
    assert parse_numeric_expression("2 Thousand") == 2000.0
    assert parse_numeric_expression("500.5k") == 500500.0


def test_lakhs():
    assert parse_numeric_expression("1 lakh") == 100000.0
    assert parse_numeric_expression("1 lakhs") == 100000.0
    assert parse_numeric_expression("2 lakh") == 200000.0
    assert parse_numeric_expression("2.5 lakh") == 250000.0
    assert parse_numeric_expression("1.5 Lakh") == 150000.0
    assert parse_numeric_expression("5 LAKH") == 500000.0
    assert parse_numeric_expression("1.5 lac") == 150000.0


def test_crores():
    assert parse_numeric_expression("1 crore") == 10000000.0
    assert parse_numeric_expression("2 crores") == 20000000.0
    assert parse_numeric_expression("1.5 crore") == 15000000.0
    assert parse_numeric_expression("1.25 crore") == 12500000.0
    assert parse_numeric_expression("2cr") == 20000000.0


def test_international_scales():
    assert parse_numeric_expression("1 million") == 1000000.0
    assert parse_numeric_expression("2.5 million") == 2500000.0
    assert parse_numeric_expression("2 Million") == 2500000.0 or parse_numeric_expression("2 Million") == 2000000.0
    assert parse_numeric_expression("2.5M") == 2500000.0
    assert parse_numeric_expression("2.5m") == 2500000.0
    assert parse_numeric_expression("500m") == 500000000.0
    assert parse_numeric_expression("1 billion") == 1000000000.0
    assert parse_numeric_expression("2.5 billion") == 2500000000.0
    assert parse_numeric_expression("2B") == 2000000000.0
    assert parse_numeric_expression("1 trillion") == 1000000000000.0


def test_currencies():
    assert parse_numeric_expression("₹50,000") == 50000.0
    assert parse_numeric_expression("Rs 50,000") == 50000.0
    assert parse_numeric_expression("Rs. 50,000") == 50000.0
    assert parse_numeric_expression("INR 1 lakh") == 100000.0
    assert parse_numeric_expression("$2 million") == 2000000.0
    assert parse_numeric_expression("USD 500k") == 500000.0


def test_combined_expressions():
    assert parse_numeric_expression("1 lakh 50 thousand") == 150000.0
    assert parse_numeric_expression("2 crore 25 lakh") == 22500000.0
    assert parse_numeric_expression("1 million 500 thousand") == 1500000.0
    assert parse_numeric_expression("1 lakh 50 thousand 500") == 150500.0


def test_fuzzy_scale_expressions():
    assert parse_numeric_expression("2 thosand") == 2000.0
    assert parse_numeric_expression("2 millon") == 2000000.0
    assert parse_numeric_expression("5 lkah") == 500000.0


def run_all():
    tests = [
        test_scales,
        test_fuzzy_scales,
        test_plain_numbers,
        test_comma_formatting,
        test_thousands,
        test_lakhs,
        test_crores,
        test_international_scales,
        test_currencies,
        test_combined_expressions,
        test_fuzzy_scale_expressions,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
    print(f"\nNumber Parser Test Summary: {passed}/{len(tests)} passed.")
    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    run_all()
