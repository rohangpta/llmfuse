"""
LLMFuse - LLM-based compression using arithmetic coding.

This package implements arithmetic coding compression that leverages language model
probability distributions for efficient text compression.
"""

from .arithmetic_coding import ArithmeticCoder, estimate_compression_ratio
from .llmencode import LLMEncode
from .model import (
    get_model_response,
    get_model_logprobs,
    compress_with_model_probs,
    decompress_with_model_probs,
    query_model,
    validate_api_key,
    get_available_models,
    get_available_qwen3_models,
    print_available_models,
)

__version__ = "0.1.0"
__author__ = "LLMFuse Team"
__description__ = "LLM-based text compression using arithmetic coding"

__all__ = [
    # Main compression interface
    "LLMEncode",
    
    # Arithmetic coding
    "ArithmeticCoder",
    "estimate_compression_ratio",
    
    # Model interface
    "get_model_response",
    "get_model_logprobs", 
    "compress_with_model_probs",
    "decompress_with_model_probs",
    "query_model",
    "validate_api_key",
    "get_available_models",
    "get_available_qwen3_models",
    "print_available_models",
] 