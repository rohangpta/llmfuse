"""
LLMFuse - LLM-based compression using arithmetic coding.

This package implements arithmetic coding compression that leverages language model
probability distributions for efficient text compression.
"""

# Core modules remain in src
# LLMEncode functionality moved to llmencode package
# LLMFS functionality moved to llmfuse package
from .model import (
    get_model_response,
    get_model_logprobs,
    compress_with_model_probs,
    decompress_with_model_probs,
    query_model,
    validate_api_key,
    get_available_models,
    get_available_qwen3_models,
)

__version__ = "0.1.0"
__author__ = "LLMFuse Team"
__description__ = "LLM-based text compression using arithmetic coding"

__all__ = [
    # Model interface
    "get_model_response",
    "get_model_logprobs", 
    "compress_with_model_probs",
    "decompress_with_model_probs",
    "query_model",
    "validate_api_key",
    "get_available_models",
    "get_available_qwen3_models",
] 