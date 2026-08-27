import re
from rapidfuzz import fuzz, distance

def _soundex(word: str) -> str:
    w = word.upper()
    if not w:
        return ""
    code_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    first_letter = w[0]
    codes = []
    prev_code = code_map.get(first_letter, '')
    for char in w[1:]:
        c = code_map.get(char, '')
        if c and c != prev_code:
            codes.append(c)
        prev_code = c
    return (first_letter + "".join(codes) + "0000")[:4]

def _consonant_skel(w: str) -> str:
    w = w.lower()
    if not w: return ""
    return w[0] + re.sub(r'[aeiouy]', '', w[1:])

def _collapse_repeats(w: str) -> str:
    return re.sub(r'(.)\1+', r'\1', w.lower())

test_pairs = [
    ("nykks", "nike"),
    ("shoss", "shoes"),
    ("nyke", "nike"),
    ("shose", "shoes"),
    ("samsng", "samsung"),
    ("phne", "phone"),
    ("wirless", "wireless"),
    ("hedphnes", "headphones"),
    ("blutooth", "bluetooth"),
    ("hedphones", "headphones"),
    ("lptap", "laptop"),
    ("backpac", "backpack"),
]

for raw, target in test_pairs:
    col = _collapse_repeats(raw)
    dist_raw = distance.Levenshtein.distance(raw, target)
    dist_col = distance.Levenshtein.distance(col, target)
    best_dist = min(dist_raw, dist_col)
    r_raw = fuzz.ratio(raw, target)
    r_col = fuzz.ratio(col, target)
    best_r = max(r_raw, r_col)
    jw_raw = distance.JaroWinkler.similarity(raw, target)
    jw_col = distance.JaroWinkler.similarity(col, target)
    best_jw = max(jw_raw, jw_col)
    sx_raw = _soundex(raw)
    sx_col = _soundex(col)
    sx_tgt = _soundex(target)
    phonetic = (sx_raw == sx_tgt) or (sx_col == sx_tgt)
    skel_dist = distance.Levenshtein.distance(_consonant_skel(raw), _consonant_skel(target))
    print(f"{raw!r:12} -> {target!r:12} | dist:{dist_raw}->{best_dist} | ratio:{best_r:.1f} | JW:{best_jw:.3f} | soundex:{phonetic} | skel_dist:{skel_dist}")
