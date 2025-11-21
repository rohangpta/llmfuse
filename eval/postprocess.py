import re
from typing import Tuple

from llmfuse.utils import STATE_STOP_TOKEN

EXCLUDED_PREFIXES = ("<think",)


def strip_think(text: str) -> str:
    if not text:
        return text
    # Remove closed or unclosed <think> blocks
    text = re.sub(r"<think>[\s\S]*?(</think>|$)", "", text, flags=re.IGNORECASE)
    # Remove leading think-like lines
    lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith("<think")]
    return "\n".join(lines).strip()


def is_state_expected(expected: str) -> bool:
    """Detect whether the expected output is a filesystem tree/XML blob."""
    if not expected:
        return False
    stripped = expected.lstrip()
    return stripped.startswith('<filesystem') or stripped.startswith('/')


def _extract_xml_state(predicted: str) -> str | None:
    if not predicted:
        return None
    text = predicted.strip()
    start = text.find('<filesystem')
    if start == -1:
        return None
    end = text.find('</filesystem>', start)
    if end == -1:
        return text[start:].strip()
    end += len('</filesystem>')
    return text[start:end].strip()


def sanitize_state_output(predicted: str) -> str:
    if not predicted:
        return predicted
    
    xml_block = _extract_xml_state(predicted)
    if xml_block is not None:
        return xml_block
    return predicted.strip()


def strip_state_stop_token(text: str | None) -> str | None:
    if not text:
        return text
    stripped = text.rstrip()
    if stripped.endswith(STATE_STOP_TOKEN):
        stripped = stripped[: -len(STATE_STOP_TOKEN)].rstrip()
    return stripped


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
    parts = re.split(r"(\[EOF\]|^Answer:|^Explanation:)", predicted, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)
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
    if is_state_expected(expected):
        cleaned_expected = strip_state_stop_token(expected) if expected else expected
        cleaned_predicted = strip_state_stop_token(sanitize_state_output(predicted))
        return clamp_to_expected(cleaned_predicted, cleaned_expected)

    if expected and expected.strip().startswith("["):
        cleaned = sanitize_readdir_output(predicted, expected)
        return clamp_to_expected(cleaned, expected)

    # Default: just clamp
    return clamp_to_expected(predicted, expected)
