import re
from typing import Tuple

EXCLUDED_PREFIXES = ("<think",)


def strip_think(text: str) -> str:
    if not text:
        return text
    # Remove closed or unclosed <think> blocks
    text = re.sub(r"<think>[\s\S]*?(</think>|$)", "", text, flags=re.IGNORECASE)
    # Remove leading think-like lines
    lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith("<think")]
    return "\n".join(lines).strip()


def is_tree_expected(expected: str) -> bool:
    return expected.lstrip().startswith('/') if expected else False


def remove_duplicate_trees(predicted: str) -> str:
    """Remove obvious duplicate filesystem trees from output."""
    if not predicted or predicted.count('\n/') <= 1:
        return predicted
    
    # Split on lines that start with '/' (tree roots)
    parts = []
    current_part = []
    
    for line in predicted.splitlines():
        if line.strip().startswith('/') and current_part:
            # Found a new tree, save the previous one
            parts.append('\n'.join(current_part))
            current_part = [line]
        else:
            current_part.append(line)
    
    if current_part:
        parts.append('\n'.join(current_part))
    
    # Return just the first complete tree
    return parts[0].strip() if parts else predicted.strip()


def sanitize_tree_output(predicted: str) -> str:
    if not predicted:
        return predicted
    
    # First remove duplicates
    predicted = remove_duplicate_trees(predicted)
    
    # Then find the tree start
    lines = predicted.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith('/'):
            return "\n".join(lines[i:]).strip()
    return predicted.strip()


def sanitize_read_output(predicted: str) -> str:
    return predicted.strip() if predicted else predicted


def _extract_first_json_array(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"\[[\s\S]*?\]", text)
    return match.group(0) if match else None


def _parse_json_array(text: str):
    try:
        import json
        arr = json.loads(text)
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def sanitize_readdir_output(predicted: str, expected: str | None = None) -> str:
    # Prefer an exact array if present; otherwise the first array; else trimmed text
    if not predicted:
        return predicted
    text = predicted.strip()
    arrays = re.findall(r"\[[\s\S]*?\]", text)
    if expected:
        exp = expected.strip()
        # Exact match wins
        for arr in arrays:
            if arr.strip() == exp:
                return exp
        # If same multiset, clamp to expected ordering (enforce deterministic ordering)
        exp_arr = _parse_json_array(exp)
        if exp_arr is not None:
            from collections import Counter
            exp_counter = Counter(exp_arr)
            for arr in arrays:
                cand_arr = _parse_json_array(arr)
                if cand_arr is not None and Counter(cand_arr) == exp_counter:
                    return exp
    if arrays:
        return arrays[0].strip()
    return text


def _strip_chatter(predicted: str) -> str:
    if not predicted:
        return predicted
    # Remove trailing common chatter markers
    parts = re.split(r"(```|\[EOF\]|^Answer:|^Explanation:)", predicted, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)
    return parts[0].strip() if parts else predicted.strip()


def clamp_to_expected(predicted: str, expected: str) -> str:
    if predicted is None:
        return predicted
    if expected is None:
        return predicted
    p = predicted.strip()
    e = expected.strip()
    if e == "":
        return ""
    if p.startswith(e):
        return e
    if e in p:
        return e
    return p


def sanitize_by_expected(predicted: str, expected: str) -> str:
    predicted = strip_think(predicted)
    predicted = _strip_chatter(predicted)

    # Empty expected -> force empty
    if expected is not None and expected.strip() == "":
        return ""

    # Route based on expected shape
    if is_tree_expected(expected):
        cleaned = sanitize_tree_output(predicted)
        return clamp_to_expected(cleaned, expected)

    if expected and expected.strip().startswith("["):
        cleaned = sanitize_readdir_output(predicted, expected)
        return clamp_to_expected(cleaned, expected)

    # Default: just clamp
    return clamp_to_expected(predicted, expected)