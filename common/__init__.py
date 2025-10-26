"""
LLMFuse Common - Shared utilities and model interfaces.

This package contains common functionality used across the LLM-FUSE project:
- Model interfaces for LLM communication
- Shared utilities and helper functions
"""

# Core exports (trimmed build)
from .model import (
    get_model_response,
    get_model_logprobs,
    compress_with_model_probs,
    decompress_with_model_probs,
    get_model,
    get_tokenizer,
    get_qwen3_4b_id,
)

__version__ = "0.1.0"
__author__ = "LLMFuse Team"
__description__ = "Common utilities and model interfaces for LLM-FUSE"

__all__ = [
    # Model helpers
    "get_model_response",
    "get_model_logprobs",
    "compress_with_model_probs",
    "decompress_with_model_probs",
    "get_model",
    "get_tokenizer",
    "get_qwen3_4b_id",
]
