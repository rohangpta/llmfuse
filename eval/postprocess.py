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


def sanitize_tree_output(predicted: str) -> str:
    if not predicted:
        return predicted
    lines = predicted.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith('/'):
            return "\n".join(lines[i:]).strip()
    return predicted.strip()


def sanitize_read_output(predicted: str) -> str:
    return predicted.strip() if predicted else predicted


def sanitize_readdir_output(predicted: str) -> str:
    # Assume expected is a JSON-like array; do not force tree trimming
    return predicted.strip() if predicted else predicted


def sanitize_by_expected(predicted: str, expected: str) -> str:
    predicted = strip_think(predicted)
    if is_tree_expected(expected):
        return sanitize_tree_output(predicted)
    return predicted 