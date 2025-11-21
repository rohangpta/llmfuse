from typing import Dict


def exact_match(predicted: str, expected: str) -> bool:
    if predicted is None or expected is None:
        return False
    return predicted.strip() == expected.strip()


def simple_similarity(predicted: str, expected: str) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    # Quick exact/normalized match checks
    if predicted.strip() == expected.strip():
        return 1.0

    pred_norm = " ".join(predicted.split())
    exp_norm = " ".join(expected.split())
    if pred_norm == exp_norm:
        return 1.0

    # Character set Jaccard similarity as fallback
    pred_chars = set(pred_norm.lower())
    exp_chars = set(exp_norm.lower())
    if not pred_chars and not exp_chars:
        return 1.0
    inter = len(pred_chars.intersection(exp_chars))
    union = len(pred_chars.union(exp_chars))
    return inter / union if union else 0.0 