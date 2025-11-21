"""
Minimal model utilities for the trimmed LLM-FUSE pipeline.

Only Qwen3-4B is supported. The helpers below provide just enough surface area
for the Modal training/evaluation scripts and the archived LLMEncode playground.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmfuse.utils import STATE_STOP_TOKEN

QWEN3_4B_SHORT_NAME = "qwen3-4b"
QWEN3_4B_HF_ID = "Qwen/Qwen3-4B"
MAX_GENERATION_TOKENS = 1024


def get_qwen3_4b_id() -> str:
    """Return the canonical Hugging Face identifier for Qwen3-4B."""
    return QWEN3_4B_HF_ID


def _resolve_model_name(model_name: str) -> str:
    if model_name.lower() in (QWEN3_4B_SHORT_NAME, QWEN3_4B_HF_ID.lower()):
        return QWEN3_4B_HF_ID
    return model_name


@lru_cache(maxsize=1)
def _load_tokenizer(model_name: str) -> AutoTokenizer:
    resolved = _resolve_model_name(model_name)
    return AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> AutoModelForCausalLM:
    resolved = _resolve_model_name(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def get_model_response(
    prompt: str,
    model_name: str = QWEN3_4B_SHORT_NAME,
    temperature: float = 0.0,
    max_output_tokens: int = MAX_GENERATION_TOKENS,
    **generate_kwargs: Any,
) -> str:
    """
    Generate a deterministic response from Qwen3-4B.

    The interface mirrors the legacy helper but only supports the trimmed setup.
    """
    tokenizer = _load_tokenizer(model_name)
    model = _load_model(model_name)
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if STATE_STOP_TOKEN in text:
        text = text.split(STATE_STOP_TOKEN, 1)[0].strip()
    return text


def get_model_logprobs(
    context: str,
    model: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    verbose: bool = False,
) -> List[float]:
    """
    Retrieve the next-token probability distribution for the given context.
    """
    if tokenizer is None:
        tokenizer = _load_tokenizer(QWEN3_4B_SHORT_NAME)
    if model is None:
        model = _load_model(QWEN3_4B_SHORT_NAME)

    device = next(model.parameters()).device
    input_tokens = tokenizer.encode(context, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(input_tokens).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    prob_floor = 1e-15
    probs = np.maximum(probs, prob_floor)
    probs /= probs.sum()

    if verbose:
        top_indices = np.argsort(probs)[-5:][::-1]
        top_tokens = [(idx, float(probs[idx]), repr(tokenizer.decode([idx]))) for idx in top_indices]
        print(f"[DEBUG] Top predictions: {top_tokens}")

    return probs.tolist()


def get_tokenizer(model_name: str = QWEN3_4B_SHORT_NAME) -> AutoTokenizer:
    return _load_tokenizer(model_name)


def get_model(model_name: str = QWEN3_4B_SHORT_NAME) -> AutoModelForCausalLM:
    return _load_model(model_name)


def compress_with_model_probs(
    text: str,
    model_name: str = QWEN3_4B_SHORT_NAME,
    max_tokens: int = 1000,
    verbose: bool = False,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Compress text using arithmetic coding guided by the model's probabilities.
    """
    tokenizer = get_tokenizer(model_name)
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(input_ids) > max_tokens:
        raise ValueError(f"Input has {len(input_ids)} tokens, exceeds max_tokens={max_tokens}")

    model = get_model(model_name)
    probs_per_token: List[List[float]] = []

    start_token = "<START>"
    for i, token_id in enumerate(input_ids):
        context = start_token if i == 0 else start_token + tokenizer.decode(input_ids[:i], skip_special_tokens=True)
        probs = get_model_logprobs(context, model=model, tokenizer=tokenizer, verbose=verbose)
        probs_per_token.append(probs)

    from llmencode.arithmetic_coding import ArithmeticCoder

    coder = ArithmeticCoder()
    compressed_bytes = coder.encode(input_ids, probs_per_token)

    metadata = {
        "original_length": len(text),
        "num_tokens": len(input_ids),
        "model_name": model_name,
        "start_token": start_token,
        "compressed_length": len(compressed_bytes),
    }

    return compressed_bytes, metadata


def decompress_with_model_probs(
    compressed_bytes: bytes,
    metadata: Dict[str, Any],
    verbose: bool = False,
) -> str:
    """
    Decompress bytes produced by `compress_with_model_probs`.
    """
    model_name = metadata["model_name"]
    num_tokens = metadata["num_tokens"]
    start_token = metadata["start_token"]

    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)

    from llmencode.arithmetic_coding import ArithmeticCoder

    coder = ArithmeticCoder()
    coder.decode_iterative_init(compressed_bytes)

    decoded_ids: List[int] = []
    for i in range(num_tokens):
        context = start_token if i == 0 else start_token + tokenizer.decode(decoded_ids, skip_special_tokens=True)
        probs = get_model_logprobs(context, model=model, tokenizer=tokenizer, verbose=verbose)
        token_id = coder.decode_iterative_next(probs)
        decoded_ids.append(token_id)

    return tokenizer.decode(decoded_ids, skip_special_tokens=True)
