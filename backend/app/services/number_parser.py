"""
Generic Numeric Expression Parser for Price Constraints

Architecture & Design:
----------------------
1. Data-Driven Scale Multipliers:
   - Indian system: lakh/lakhs/lac (10^5), crore/crores/cr (10^7)
   - Western system: thousand/k (10^3), million/m (10^6), billion/b (10^9), trillion/t (10^12)
   - Case-insensitive, singular and plural forms supported.
2. Multi-Unit Combined Expressions:
   - Evaluates additive multi-unit expressions (e.g., '1 lakh 50 thousand' -> 150,000,
     '2 crore 25 lakh' -> 22,500,000, '1 million 500 thousand' -> 1,500,000).
3. Robust Decimal & Comma Formatting:
   - Handles standard, Western ('1,000,000'), and Indian ('1,00,000') comma groupings.
   - Preserves floating point precision for decimal scales ('1.5 lakh', '2.5 million', '500.5k').
4. Currency Agnostic:
   - Cleans currency symbols and prefixes/suffixes (₹, $, Rs, Rs., INR, USD).
5. High-Confidence Typo-Tolerant Scale Recognition:
   - Recovers misspelled full scale words (e.g. 'thosand' -> thousand, 'millon' -> million, 'lkah' -> lakh)
   - Strict gating: short abbreviations ('k', 'm', 'b', 't', 'cr') must match exactly.
6. 100% Offline with zero cloud/network dependencies.
"""

import re
from typing import Dict, List, Optional, Tuple
from rapidfuzz import distance, fuzz

# Canonical scale multipliers
SCALE_MULTIPLIERS: Dict[str, float] = {
    # Western / International
    "k": 1_000.0,
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "b": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "trillions": 1_000_000_000_000.0,

    # Indian Numbering System
    "lakh": 100_000.0,
    "lakhs": 100_000.0,
    "lac": 100_000.0,
    "lacs": 100_000.0,
    "crore": 10_000_000.0,
    "crores": 10_000_000.0,
    "cr": 10_000_000.0,
}

# Canonical dictionary for fuzzy scale word resolution (length >= 4 only)
CANONICAL_SCALE_WORDS: Dict[str, float] = {
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "trillions": 1_000_000_000_000.0,
    "lakh": 100_000.0,
    "lakhs": 100_000.0,
    "lac": 100_000.0,
    "lacs": 100_000.0,
    "crore": 10_000_000.0,
    "crores": 10_000_000.0,
}

# Currency regex pattern for stripping
CURRENCY_PATTERN = re.compile(r"\b(?:rupees?|dollars?|bucks?|euros?|inr|usd|gbp|cents?|rs\.?)\b|[₹$€£]", re.IGNORECASE)


def resolve_scale(scale_token: str) -> Optional[float]:
    """Resolve a scale token (word or abbreviation) to its numeric multiplier.

    Supports exact lookup for short abbreviations ('k', 'm', 'b', 't', 'cr')
    and high-confidence fuzzy matching for full scale words ('thousand', 'million', 'lakh', 'crore').
    """
    cleaned = scale_token.strip().lower().rstrip(".,;:")
    if not cleaned:
        return None

    # 1. Exact match in scale table
    if cleaned in SCALE_MULTIPLIERS:
        return SCALE_MULTIPLIERS[cleaned]

    # 2. Strict guard: short abbreviations (<= 2 chars) MUST match exactly
    if len(cleaned) <= 2:
        return None

    # 3. High-confidence fuzzy match for words >= 3 chars
    best_scale: Optional[float] = None
    best_score = 0.0

    for word, multiplier in CANONICAL_SCALE_WORDS.items():
        dist = distance.DamerauLevenshtein.distance(cleaned, word)
        ratio = float(fuzz.ratio(cleaned, word))

        # Accept if 1 edit distance or high similarity ratio (>= 75 for short, >= 80 for longer)
        min_ratio = 75.0 if len(word) <= 4 else 80.0
        if dist <= 1 or ratio >= min_ratio:
            if ratio > best_score:
                best_score = ratio
                best_scale = multiplier

    return best_scale


def clean_currency_and_separators(text: str) -> str:
    """Strip currency symbols/names and commas from text."""
    cleaned = CURRENCY_PATTERN.sub(" ", text)
    cleaned = cleaned.replace(",", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_single_chunk(chunk_text: str) -> Optional[Tuple[float, str]]:
    """Parse a single numeric chunk like '2.5 million', '50k', '1.5M', or '3000'.

    Returns (chunk_value, scale_name) or None.
    """
    raw = chunk_text.strip().lower()
    if not raw:
        return None

    # 1. Check attached abbreviation/word: e.g. "50k", "1.5m", "2.5b", "500.5k", "2cr", "10lakh"
    m_attached = re.match(r"^([\d]+(?:\.[\d]+)?)\s*([a-zA-Z]+)$", raw)
    if m_attached:
        num_part = float(m_attached.group(1))
        scale_part = m_attached.group(2)
        multiplier = resolve_scale(scale_part)
        if multiplier is not None:
            return (num_part * multiplier, scale_part)
        return None

    # 2. Check plain number: e.g. "3000", "500.5"
    m_plain = re.match(r"^[\d]+(?:\.[\d]+)?$", raw)
    if m_plain:
        return (float(raw), "unit")

    return None


def parse_numeric_expression(text: str) -> Optional[float]:
    """Parse a generic numeric expression into a normalized float value.

    Examples:
    - "3000" -> 3000.0
    - "3,000" -> 3000.0
    - "2.5 million" -> 2500000.0
    - "2 lakh" -> 200000.0
    - "2 lakhs" -> 200000.0
    - "1.5 crore" -> 15000000.0
    - "500.5k" -> 500500.0
    - "1.5M" -> 1500000.0
    - "2B" -> 2000000000.0
    - "₹50,000" -> 50000.0
    - "INR 1 lakh" -> 100000.0
    - "1 lakh 50 thousand" -> 150000.0
    - "2 crore 25 lakh" -> 22500000.0
    - "1 million 500 thousand" -> 1500000.0
    - "2 thosand" -> 2000.0
    - "5 lkah" -> 500000.0

    Returns None if text does not represent a valid numeric expression.
    """
    if not text:
        return None

    cleaned = clean_currency_and_separators(text)
    if not cleaned:
        return None

    tokens = cleaned.split()
    if not tokens:
        return None

    total_value = 0.0
    tokens_consumed = 0
    idx = 0
    n = len(tokens)

    while idx < n:
        tok = tokens[idx]

        # Case A: Attached number + scale (e.g. "50k", "1.5M", "2B", "500m")
        single_res = parse_single_chunk(tok)
        if single_res is not None:
            val, scale_name = single_res
            if scale_name != "unit":
                total_value += val
                tokens_consumed += 1
                idx += 1
                continue
            # If plain number with no attached scale, check if NEXT token is a scale word
            if idx + 1 < n:
                next_tok = tokens[idx + 1]
                scale_mult = resolve_scale(next_tok)
                if scale_mult is not None:
                    total_value += val * scale_mult
                    tokens_consumed += 2
                    idx += 2
                    continue
            # Otherwise plain number
            total_value += val
            tokens_consumed += 1
            idx += 1
            continue

        # If token could not be parsed as any numeric chunk, fail
        return None

    if tokens_consumed > 0:
        return total_value

    return None
